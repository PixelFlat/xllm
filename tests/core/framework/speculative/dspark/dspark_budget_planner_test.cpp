/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/framework/speculative/dspark/dspark_budget_planner.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <vector>

namespace xllm {
namespace {

DSparkBudgetOptions profile_options() {
  DSparkBudgetOptions options;
  options.fixed_draft_time_ms = 1.0;
  options.commit_time_ms = 0.5;
  return options;
}

const std::array<DSparkGraphCost, 4> kCosts = {DSparkGraphCost{2, 1.0},
                                               {4, 2.0},
                                               {8, 4.0},
                                               {16, 8.0}};

const std::array<DSparkGraphCost, 9> kCalibrationGraphs = {
    DSparkGraphCost{1, 0.0},
    {2, 0.0},
    {4, 0.0},
    {8, 0.0},
    {16, 0.0},
    {32, 0.0},
    {48, 0.0},
    {64, 0.0},
    {128, 0.0}};

TEST(DSparkBudgetPlannerTest, SupportsConfiguredDraftWidthAtCommonBatchSizes) {
  for (const uint32_t draft_width : {5U, 7U}) {
    for (const uint32_t batch_size : {1U, 8U, 16U}) {
      SCOPED_TRACE(draft_width);
      SCOPED_TRACE(batch_size);
      auto options = profile_options();
      options.draft_width = draft_width;
      const std::array<DSparkGraphCost, 3> costs = {
          DSparkGraphCost{batch_size, 1.0},
          {batch_size * 2, 1.5},
          {batch_size * (draft_width + 1), 8.0}};
      const auto full =
          DSparkBudgetPlanner::plan(batch_size, {}, costs, options);
      ASSERT_TRUE(full);
      EXPECT_EQ(full->draft_budget, batch_size * draft_width);
      EXPECT_EQ(full->graph_rows, batch_size * (draft_width + 1));
      const std::vector<double> confidence(batch_size * draft_width, 0.0);
      const auto pruned =
          DSparkBudgetPlanner::plan(batch_size, confidence, costs, options);
      ASSERT_TRUE(pruned);
      EXPECT_EQ(pruned->draft_budget, 0U);
      EXPECT_EQ(pruned->graph_rows, batch_size);
      const auto calibration = DSparkBudgetPlanner::calibration_plan(
          batch_size, batch_size, costs.back().graph_rows, costs, draft_width);
      ASSERT_TRUE(calibration);
      EXPECT_EQ(calibration->draft_budget, batch_size * draft_width);
    }
  }
}

TEST(DSparkBudgetPlannerTest, RejectsUnsupportedWidthsAndMismatchedFeedback) {
  auto options = profile_options();
  for (const uint32_t draft_width : {0U, 3U, 6U, 8U}) {
    options.draft_width = draft_width;
    EXPECT_FALSE(DSparkBudgetPlanner::plan(1, {}, kCosts, options));
    EXPECT_FALSE(
        DSparkBudgetPlanner::calibration_plan(1, 1, 8, kCosts, draft_width));
  }
  options.draft_width = 5;
  EXPECT_FALSE(DSparkBudgetPlanner::plan(
      1, std::vector<double>(7, 0.5), kCosts, options));
  options.draft_width = 7;
  EXPECT_FALSE(DSparkBudgetPlanner::plan(
      1, std::vector<double>(5, 0.5), kCosts, options));
}

TEST(DSparkBudgetPlannerTest, CalibrationUsesKnownTierWithoutCostMeasurements) {
  const auto plan = DSparkBudgetPlanner::calibration_plan(
      /*logical_requests=*/8,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/16,
      kCalibrationGraphs);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->graph_rows, 16U);
  EXPECT_EQ(plan->draft_budget, 8U);
}

TEST(DSparkBudgetPlannerTest, CalibrationAllowsSmallTransientRequestBuckets) {
  const auto plan = DSparkBudgetPlanner::calibration_plan(
      /*logical_requests=*/1,
      /*physical_requests=*/1,
      /*requested_graph_rows=*/64,
      kCalibrationGraphs);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->graph_rows, 8U);
  EXPECT_EQ(plan->draft_budget, 7U);
}

TEST(DSparkBudgetPlannerTest, CalibrationAllowsLargerTransientRequestBuckets) {
  const auto plan = DSparkBudgetPlanner::calibration_plan(
      /*logical_requests=*/16,
      /*physical_requests=*/16,
      /*requested_graph_rows=*/8,
      kCalibrationGraphs);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->graph_rows, 128U);
  EXPECT_EQ(plan->draft_budget, 112U);
}

TEST(DSparkBudgetPlannerTest, CalibrationUsesLogicalBudgetAndPhysicalCapacity) {
  const auto plan = DSparkBudgetPlanner::calibration_plan(
      /*logical_requests=*/3,
      /*physical_requests=*/4,
      /*requested_graph_rows=*/4,
      kCalibrationGraphs);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->graph_rows, 4U);
  EXPECT_EQ(plan->draft_budget, 1U);
  const auto transient = DSparkBudgetPlanner::calibration_plan(
      /*logical_requests=*/3,
      /*physical_requests=*/3,
      /*requested_graph_rows=*/64,
      kCalibrationGraphs);
  ASSERT_TRUE(transient);
  EXPECT_EQ(transient->graph_rows, 32U);
  EXPECT_EQ(transient->draft_budget, 21U);
}

TEST(DSparkBudgetPlannerTest, CalibrationSupportsAnchorOnlyVerification) {
  const auto plan = DSparkBudgetPlanner::calibration_plan(
      /*logical_requests=*/8,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/8,
      kCalibrationGraphs);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->graph_rows, 8U);
  EXPECT_EQ(plan->draft_budget, 0U);
}

TEST(DSparkBudgetPlannerTest, CalibrationUsesWholeRoundBindingForPartialBatch) {
  const std::array<DSparkGraphCost, 4> actual = {
      DSparkGraphCost{8, 0.0}, {16, 0.0}, {32, 0.0}, {48, 0.0}};
  const auto transient = DSparkBudgetPlanner::calibration_plan(
      /*logical_requests=*/5,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/64,
      kCalibrationGraphs,
      /*draft_width=*/7,
      actual);
  ASSERT_TRUE(transient);
  EXPECT_EQ(transient->graph_rows, 48U);
  EXPECT_EQ(transient->draft_budget, 35U);
  const auto available = DSparkBudgetPlanner::calibration_plan(
      /*logical_requests=*/5,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/16,
      kCalibrationGraphs,
      /*draft_width=*/7,
      actual);
  ASSERT_TRUE(available);
  EXPECT_EQ(available->graph_rows, 16U);
  EXPECT_EQ(available->draft_budget, 11U);
}

TEST(DSparkBudgetPlannerTest, CalibrationRejectsStableMissingWholeRoundTier) {
  const std::array<DSparkGraphCost, 4> actual = {
      DSparkGraphCost{8, 0.0}, {16, 0.0}, {32, 0.0}, {64, 0.0}};
  EXPECT_FALSE(
      DSparkBudgetPlanner::calibration_plan(/*logical_requests=*/8,
                                            /*physical_requests=*/8,
                                            /*requested_graph_rows=*/48,
                                            kCalibrationGraphs,
                                            /*draft_width=*/7,
                                            actual));
  EXPECT_FALSE(
      DSparkBudgetPlanner::calibration_plan(/*logical_requests=*/5,
                                            /*physical_requests=*/8,
                                            /*requested_graph_rows=*/64,
                                            kCalibrationGraphs,
                                            /*draft_width=*/7,
                                            std::span(actual).first(3)));
  const std::array<DSparkGraphCost, 2> unknown = {DSparkGraphCost{8, 0.0},
                                                  {40, 0.0}};
  EXPECT_FALSE(
      DSparkBudgetPlanner::calibration_plan(/*logical_requests=*/5,
                                            /*physical_requests=*/8,
                                            /*requested_graph_rows=*/64,
                                            kCalibrationGraphs,
                                            /*draft_width=*/7,
                                            unknown));
}

TEST(DSparkBudgetPlannerTest, CalibrationUsesDraftFiveWholeRoundCoverage) {
  const std::array<DSparkGraphCost, 4> actual = {
      DSparkGraphCost{8, 0.0}, {16, 0.0}, {32, 0.0}, {48, 0.0}};
  const auto plan = DSparkBudgetPlanner::calibration_plan(
      /*logical_requests=*/8,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/48,
      kCalibrationGraphs,
      /*draft_width=*/5,
      actual);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->graph_rows, 48U);
  EXPECT_EQ(plan->draft_budget, 40U);
  const auto transient = DSparkBudgetPlanner::calibration_plan(
      /*logical_requests=*/5,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/48,
      kCalibrationGraphs,
      /*draft_width=*/5,
      std::span(actual).first(3),
      /*calibration_requests=*/8);
  ASSERT_TRUE(transient);
  EXPECT_EQ(transient->graph_rows, 32U);
  EXPECT_EQ(transient->draft_budget, 25U);
}

TEST(DSparkBudgetPlannerTest,
     CalibrationRecognizesDifferentWholeRoundRequestSet) {
  const std::array<DSparkGraphCost, 4> actual = {
      DSparkGraphCost{8, 0.0}, {16, 0.0}, {32, 0.0}, {64, 0.0}};
  // A sample for B=10 can use G=48. An interim B=Bcap=8 cannot replay
  // that whole graph, but can advance at its prewarmed full-width G=64.
  const auto transient = DSparkBudgetPlanner::calibration_plan(
      /*logical_requests=*/8,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/48,
      kCalibrationGraphs,
      /*draft_width=*/7,
      actual,
      /*calibration_requests=*/10);
  ASSERT_TRUE(transient);
  EXPECT_EQ(transient->graph_rows, 64U);
  EXPECT_EQ(transient->draft_budget, 56U);
  EXPECT_FALSE(
      DSparkBudgetPlanner::calibration_plan(/*logical_requests=*/8,
                                            /*physical_requests=*/8,
                                            /*requested_graph_rows=*/48,
                                            kCalibrationGraphs,
                                            /*draft_width=*/7,
                                            actual,
                                            /*calibration_requests=*/8));
}

TEST(DSparkBudgetPlannerTest,
     CalibrationRejectsUnknownTierOrInsufficientCapacity) {
  EXPECT_FALSE(DSparkBudgetPlanner::calibration_plan(/*logical_requests=*/1,
                                                     /*physical_requests=*/1,
                                                     /*requested_graph_rows=*/3,
                                                     kCalibrationGraphs));
  EXPECT_FALSE(DSparkBudgetPlanner::calibration_plan(/*logical_requests=*/1,
                                                     /*physical_requests=*/1,
                                                     /*requested_graph_rows=*/0,
                                                     kCalibrationGraphs));
  const std::array<DSparkGraphCost, 2> insufficient = {DSparkGraphCost{8, 0.0},
                                                       {16, 0.0}};
  EXPECT_FALSE(
      DSparkBudgetPlanner::calibration_plan(/*logical_requests=*/8,
                                            /*physical_requests=*/8,
                                            /*requested_graph_rows=*/16,
                                            insufficient));
  EXPECT_FALSE(DSparkBudgetPlanner::calibration_plan(/*logical_requests=*/1,
                                                     /*physical_requests=*/1,
                                                     /*requested_graph_rows=*/8,
                                                     {}));
}

TEST(DSparkBudgetPlannerTest,
     CalibrationRejectsInvalidGeometryAndTierOrdering) {
  EXPECT_FALSE(DSparkBudgetPlanner::calibration_plan(/*logical_requests=*/0,
                                                     /*physical_requests=*/1,
                                                     /*requested_graph_rows=*/8,
                                                     kCalibrationGraphs));
  EXPECT_FALSE(DSparkBudgetPlanner::calibration_plan(/*logical_requests=*/1,
                                                     /*physical_requests=*/0,
                                                     /*requested_graph_rows=*/8,
                                                     kCalibrationGraphs));
  EXPECT_FALSE(DSparkBudgetPlanner::calibration_plan(/*logical_requests=*/2,
                                                     /*physical_requests=*/1,
                                                     /*requested_graph_rows=*/8,
                                                     kCalibrationGraphs));
  EXPECT_FALSE(DSparkBudgetPlanner::calibration_plan(
      /*logical_requests=*/1,
      /*physical_requests=*/std::numeric_limits<uint32_t>::max(),
      /*requested_graph_rows=*/8,
      kCalibrationGraphs));
  const std::array<DSparkGraphCost, 2> duplicated = {DSparkGraphCost{8, 0.0},
                                                     {8, 0.0}};
  EXPECT_FALSE(DSparkBudgetPlanner::calibration_plan(/*logical_requests=*/1,
                                                     /*physical_requests=*/1,
                                                     /*requested_graph_rows=*/8,
                                                     duplicated));
}

TEST(DSparkBudgetPlannerTest, LowConfidenceSelectsSmallerPhysicalGraph) {
  const std::vector<double> confidence(14, 0.01);
  const auto plan =
      DSparkBudgetPlanner::plan(2, confidence, kCosts, profile_options());
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 0U);
  EXPECT_EQ(plan->graph_rows, 2U);
}

TEST(DSparkBudgetPlannerTest, HighConfidenceKeepsAllDrafts) {
  const std::vector<double> confidence(14, 1.0);
  const auto plan =
      DSparkBudgetPlanner::plan(2, confidence, kCosts, profile_options());
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 14U);
  EXPECT_EQ(plan->graph_rows, 16U);
}

TEST(DSparkBudgetPlannerTest, AllocatesOneBatchBudgetFromConditionalPaths) {
  const std::vector<double> confidence{0.99,
                                       0.99,
                                       0.99,
                                       0.99,
                                       0.99,
                                       0.99,
                                       0.99,
                                       0.01,
                                       0.01,
                                       0.01,
                                       0.01,
                                       0.01,
                                       0.01,
                                       0.01};
  const std::array<DSparkGraphCost, 2> costs = {DSparkGraphCost{4, 0.1},
                                                {16, 100.0}};
  const auto plan =
      DSparkBudgetPlanner::plan(2, confidence, costs, profile_options());
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 2U);
  EXPECT_EQ(plan->graph_rows, 4U);
}

TEST(DSparkBudgetPlannerTest, FillsUnusedCapacityWithoutChangingGraphCost) {
  const std::vector<double> confidence(7, 0.01);
  const std::array<DSparkGraphCost, 2> costs = {DSparkGraphCost{1, 1.0},
                                                {8, 1.0}};
  const auto plan = DSparkBudgetPlanner::plan(
      1, confidence, costs, profile_options(), DSparkBudgetPlan{5, 8});
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 7U);
  EXPECT_EQ(plan->graph_rows, 8U);
}

TEST(DSparkBudgetPlannerTest, RoundsFullWidthUpToWarmedCapacity) {
  const std::vector<double> confidence(21, 1.0);
  const std::array<DSparkGraphCost, 2> costs = {DSparkGraphCost{8, 3.0},
                                                {32, 4.0}};
  const auto plan =
      DSparkBudgetPlanner::plan(3, confidence, costs, profile_options());
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 21U);
  EXPECT_EQ(plan->graph_rows, 32U);
}

TEST(DSparkBudgetPlannerTest, TiedScoresChooseSmallestGraphDeterministically) {
  const std::vector<double> confidence(14, 0.0);
  const std::array<DSparkGraphCost, 3> costs = {
      DSparkGraphCost{2, 1.0}, {8, 1.0}, {16, 1.0}};
  for (uint32_t repeat = 0; repeat < 5; ++repeat) {
    const auto plan =
        DSparkBudgetPlanner::plan(2, confidence, costs, profile_options());
    ASSERT_TRUE(plan.has_value());
    EXPECT_EQ(plan->draft_budget, 0U);
    EXPECT_EQ(plan->graph_rows, 2U);
  }
}

TEST(DSparkBudgetPlannerTest, FillsSpareRowsEvenWhenPathProbabilityIsZero) {
  const std::vector<double> confidence(14, 0.0);
  const std::array<DSparkGraphCost, 1> costs = {DSparkGraphCost{16, 1.0}};
  const auto plan = DSparkBudgetPlanner::plan(
      2, confidence, costs, profile_options(), DSparkBudgetPlan{5, 16});
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 14U);
  EXPECT_EQ(plan->graph_rows, 16U);
}

TEST(DSparkBudgetPlannerTest, IncludesFixedDraftAndCommitTimeInScore) {
  const std::vector<double> confidence(7, 0.3);
  const std::array<DSparkGraphCost, 2> costs = {DSparkGraphCost{2, 1.0},
                                                {8, 2.0}};
  DSparkBudgetOptions options;
  options.fixed_draft_time_ms = 0.001;
  auto plan = DSparkBudgetPlanner::plan(1, confidence, costs, options);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 1U);
  options.commit_time_ms = 100.0;
  plan = DSparkBudgetPlanner::plan(1, confidence, costs, options);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 7U);
}

TEST(DSparkBudgetPlannerTest,
     HysteresisKeepsPreviousPlanUntilGainIsSufficient) {
  const std::vector<double> confidence(7, 0.5);
  const std::array<DSparkGraphCost, 2> costs = {DSparkGraphCost{2, 1.0},
                                                {8, 2.0}};
  DSparkBudgetOptions options;
  options.fixed_draft_time_ms = 1.0;
  options.min_gain = 0.2;
  const DSparkBudgetPlan previous{7, 8};
  auto plan =
      DSparkBudgetPlanner::plan(1, confidence, costs, options, previous);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 7U);
  options.min_gain = 0.0;
  plan = DSparkBudgetPlanner::plan(1, confidence, costs, options, previous);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 1U);
}

TEST(DSparkBudgetPlannerTest, ForcesPeriodicFullWidthExploration) {
  const std::vector<double> confidence(14, 0.01);
  DSparkBudgetOptions options = profile_options();
  options.exploration_interval = 3;
  auto plan = DSparkBudgetPlanner::plan(
      2, confidence, kCosts, options, std::nullopt, 2);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 0U);
  plan = DSparkBudgetPlanner::plan(
      2, confidence, kCosts, options, DSparkBudgetPlan{0, 2}, 3);
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 14U);
  EXPECT_EQ(plan->graph_rows, 16U);
}

TEST(DSparkBudgetPlannerTest, MissingFeedbackOrProfileKeepsFullWidth) {
  const auto without_feedback =
      DSparkBudgetPlanner::plan(2, {}, kCosts, profile_options());
  ASSERT_TRUE(without_feedback.has_value());
  EXPECT_EQ(without_feedback->draft_budget, 14U);
  const std::vector<double> confidence(14, 0.01);
  const std::array<DSparkGraphCost, 3> unprofiled = {
      DSparkGraphCost{2, 0.0}, {8, 0.0}, {16, 0.0}};
  const auto without_cost =
      DSparkBudgetPlanner::plan(2, confidence, unprofiled, profile_options());
  ASSERT_TRUE(without_cost.has_value());
  EXPECT_EQ(without_cost->draft_budget, 14U);
  const std::array<DSparkGraphCost, 2> missing_full_cost = {
      DSparkGraphCost{2, 1.0}, {16, 0.0}};
  const auto without_full_cost = DSparkBudgetPlanner::plan(
      2, confidence, missing_full_cost, profile_options());
  ASSERT_TRUE(without_full_cost.has_value());
  EXPECT_EQ(without_full_cost->draft_budget, 14U);
  const auto without_draft_cost =
      DSparkBudgetPlanner::plan(2, confidence, kCosts, {});
  ASSERT_TRUE(without_draft_cost.has_value());
  EXPECT_EQ(without_draft_cost->draft_budget, 14U);
}

TEST(DSparkBudgetPlannerTest, SkipsUnprofiledSmallerGraphs) {
  const std::vector<double> confidence(14, 0.01);
  const std::array<DSparkGraphCost, 2> costs = {DSparkGraphCost{2, 0.0},
                                                {16, 8.0}};
  const auto plan =
      DSparkBudgetPlanner::plan(2, confidence, costs, profile_options());
  ASSERT_TRUE(plan.has_value());
  EXPECT_EQ(plan->draft_budget, 14U);
}

TEST(DSparkBudgetPlannerTest, RejectsInsufficientWarmedCapacity) {
  const std::array<DSparkGraphCost, 1> costs = {DSparkGraphCost{8, 1.0}};
  EXPECT_FALSE(DSparkBudgetPlanner::plan(2, {}, costs, {}).has_value());
  EXPECT_FALSE(DSparkBudgetPlanner::plan(0, {}, kCosts, {}).has_value());
  EXPECT_FALSE(DSparkBudgetPlanner::plan(
                   std::numeric_limits<uint32_t>::max(), {}, kCosts, {})
                   .has_value());
}

TEST(DSparkBudgetPlannerTest, RejectsInvalidConfidence) {
  const std::vector<double> short_feedback(13, 0.5);
  EXPECT_FALSE(
      DSparkBudgetPlanner::plan(2, short_feedback, kCosts, profile_options())
          .has_value());
  for (double invalid : {-0.01,
                         1.01,
                         std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    std::vector<double> confidence(14, 0.5);
    confidence[7] = invalid;
    EXPECT_FALSE(
        DSparkBudgetPlanner::plan(2, confidence, kCosts, profile_options())
            .has_value());
  }
}

TEST(DSparkBudgetPlannerTest, RejectsInvalidBucketsAndOptions) {
  const std::array<DSparkGraphCost, 2> duplicate = {DSparkGraphCost{16, 1.0},
                                                    {16, 2.0}};
  const std::array<DSparkGraphCost, 2> unsorted = {DSparkGraphCost{16, 2.0},
                                                   {2, 1.0}};
  const std::array<DSparkGraphCost, 1> negative = {DSparkGraphCost{16, -1.0}};
  const std::array<DSparkGraphCost, 1> nonfinite = {
      DSparkGraphCost{16, std::numeric_limits<double>::infinity()}};
  EXPECT_FALSE(DSparkBudgetPlanner::plan(2, {}, {}, {}).has_value());
  EXPECT_FALSE(DSparkBudgetPlanner::plan(2, {}, duplicate, {}).has_value());
  EXPECT_FALSE(DSparkBudgetPlanner::plan(2, {}, unsorted, {}).has_value());
  EXPECT_FALSE(DSparkBudgetPlanner::plan(2, {}, negative, {}).has_value());
  EXPECT_FALSE(DSparkBudgetPlanner::plan(2, {}, nonfinite, {}).has_value());
  DSparkBudgetOptions options = profile_options();
  options.min_gain = -1.0;
  EXPECT_FALSE(DSparkBudgetPlanner::plan(2, {}, kCosts, options).has_value());
  options = profile_options();
  options.exploration_interval = 0;
  EXPECT_FALSE(DSparkBudgetPlanner::plan(2, {}, kCosts, options).has_value());
  options.fixed_draft_time_ms = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(DSparkBudgetPlanner::plan(2, {}, kCosts, options).has_value());
  options = profile_options();
  options.fixed_draft_time_ms = std::numeric_limits<double>::max();
  options.commit_time_ms = std::numeric_limits<double>::max();
  EXPECT_FALSE(DSparkBudgetPlanner::plan(2, {}, kCosts, options).has_value());
}

TEST(DSparkBudgetPlannerTest, RejectsPreviousPlanFromDifferentCapacity) {
  EXPECT_FALSE(DSparkBudgetPlanner::plan(
                   2, {}, kCosts, profile_options(), DSparkBudgetPlan{15, 16})
                   .has_value());
  EXPECT_FALSE(DSparkBudgetPlanner::plan(
                   2, {}, kCosts, profile_options(), DSparkBudgetPlan{7, 8})
                   .has_value());
  EXPECT_FALSE(DSparkBudgetPlanner::plan(
                   2, {}, kCosts, profile_options(), DSparkBudgetPlan{1, 3})
                   .has_value());
}

TEST(DSparkBudgetPlannerTest, FullStepCostsDoNotRequireSeparateDraftTiming) {
  DSparkBudgetOptions options;
  options.cost_scope = DSparkCostScope::FULL_STEP;
  const std::vector<double> confidence(14, 0.01);
  const auto plan = DSparkBudgetPlanner::plan(2, confidence, kCosts, options);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->draft_budget, 0U);
  EXPECT_EQ(plan->graph_rows, 2U);
  options.fixed_draft_time_ms = 1.0;
  EXPECT_FALSE(DSparkBudgetPlanner::plan(2, confidence, kCosts, options));
  options.fixed_draft_time_ms = 0.0;
  options.commit_time_ms = 1.0;
  EXPECT_FALSE(DSparkBudgetPlanner::plan(2, confidence, kCosts, options));
}

TEST(DSparkBudgetPlannerTest,
     FullStepKeepsExplorationAndMissingFeedbackPolicy) {
  DSparkBudgetOptions options;
  options.cost_scope = DSparkCostScope::FULL_STEP;
  options.exploration_interval = 2;
  auto plan = DSparkBudgetPlanner::plan(2, {}, kCosts, options);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->draft_budget, 14U);
  plan = DSparkBudgetPlanner::plan(
      2, std::vector<double>(14, 0.01), kCosts, options, std::nullopt, 2);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->draft_budget, 14U);
}

}  // namespace
}  // namespace xllm
