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

#include "core/framework/speculative/dspark/dspark_sps_cost_model.h"

#include <gtest/gtest.h>

#include <limits>
#include <nlohmann/json.hpp>
#include <vector>

#include "core/framework/speculative/dspark/dspark_budget_planner.h"

namespace xllm {
namespace {

nlohmann::json sps_table() {
  return {{"sample_batch_tokens", {2, 4, 8}},
          {"sample_steps_per_sec", {100.0, 50.0, 25.0}},
          {"max_batch_tokens", 16}};
}

nlohmann::json additive_table() {
  return {{"bias_seconds", 0.010},
          {"bs_probes", {1, 3}},
          {"alpha_seconds", {0.001, 0.003}},
          {"m_probes", {0, 8}},
          {"theta_seconds", {0.0, 0.008}}};
}

TEST(DSparkSpsCostModelTest, SpsUsesFloorProbeAndMilliseconds) {
  const auto model = DSparkSpsCostModel::from_json(sps_table().dump());
  ASSERT_TRUE(model);
  EXPECT_EQ(model->kind(), DSparkSpsCostModel::Kind::SPS);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 1), 10.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 2), 10.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 3), 10.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 4), 20.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 8), 40.0);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 16), 40.0);
  EXPECT_FALSE(model->step_time_ms(1, 17));
  EXPECT_FALSE(model->step_time_ms(0, 1));
  EXPECT_FALSE(model->step_time_ms(8, 4));
}

TEST(DSparkSpsCostModelTest, SpsSupportsSingleProbeAndContextBound) {
  auto table = sps_table();
  table["sample_batch_tokens"] = {1};
  table["sample_steps_per_sec"] = {25.0};
  table["max_kv_length"] = 4096;
  const auto model = DSparkSpsCostModel::from_json(table.dump());
  ASSERT_TRUE(model);
  EXPECT_EQ(model->max_kv_length(), 4096U);
  EXPECT_DOUBLE_EQ(*model->step_time_ms(2, 8), 40.0);
}

TEST(DSparkSpsCostModelTest, AdditiveInterpolatesBothAxesAndClampsEdges) {
  const auto model = DSparkSpsCostModel::from_json(additive_table().dump());
  ASSERT_TRUE(model);
  EXPECT_EQ(model->kind(), DSparkSpsCostModel::Kind::ADDITIVE);
  EXPECT_NEAR(*model->step_time_ms(2, 4), 16.0, 1e-12);
  EXPECT_NEAR(*model->step_time_ms(1, 1), 12.0, 1e-12);
  EXPECT_NEAR(*model->step_time_ms(3, 8), 21.0, 1e-12);
  EXPECT_NEAR(*model->step_time_ms(16, 128), 21.0, 1e-12);
  EXPECT_EQ(model->max_kv_length(), std::numeric_limits<uint32_t>::max());
}

TEST(DSparkSpsCostModelTest, AdditiveAcceptsNegativeOffsetsWithPositiveTotals) {
  auto table = additive_table();
  table["alpha_seconds"] = {-0.002, -0.001};
  const auto model = DSparkSpsCostModel::from_json(table.dump());
  ASSERT_TRUE(model);
  EXPECT_NEAR(*model->step_time_ms(1, 8), 16.0, 1e-12);
  table["alpha_seconds"] = {-0.020, -0.001};
  EXPECT_FALSE(DSparkSpsCostModel::from_json(table.dump()));
}

TEST(DSparkSpsCostModelTest, AdditiveSupportsSinglePointAxes) {
  auto table = additive_table();
  table["bs_probes"] = {1};
  table["alpha_seconds"] = {0.002};
  table["m_probes"] = {0};
  table["theta_seconds"] = {0.003};
  const auto model = DSparkSpsCostModel::from_json(table.dump());
  ASSERT_TRUE(model);
  EXPECT_NEAR(*model->step_time_ms(8, 64), 15.0, 1e-12);
}

TEST(DSparkSpsCostModelTest, RejectsMalformedAmbiguousOrIncompleteJson) {
  for (const auto* text :
       {"{", "[]", "null", "{}", "{\"bias_seconds\":1e999}"}) {
    EXPECT_FALSE(DSparkSpsCostModel::from_json(text)) << text;
  }
  auto table = sps_table();
  table["bias_seconds"] = 1.0;
  EXPECT_FALSE(DSparkSpsCostModel::from_json(table.dump()));
  table = additive_table();
  table.erase("theta_seconds");
  EXPECT_FALSE(DSparkSpsCostModel::from_json(table.dump()));
}

TEST(DSparkSpsCostModelTest, RejectsInvalidAxesAndValues) {
  for (const auto& invalid : nlohmann::json{nlohmann::json::array(),
                                            {2, 2, 8},
                                            {8, 4, 2},
                                            {0, 4, 8},
                                            {-1, 4, 8},
                                            {2.0, 4, 8},
                                            {true, 4, 8},
                                            {4294967296ULL, 4, 8}}) {
    auto table = sps_table();
    table["sample_batch_tokens"] = invalid;
    EXPECT_FALSE(DSparkSpsCostModel::from_json(table.dump())) << table;
  }
  for (const auto& invalid : nlohmann::json{nlohmann::json::array(),
                                            {100, 50},
                                            {0, 50, 25},
                                            {-1, 50, 25},
                                            {"100", 50, 25},
                                            {true, 50, 25},
                                            {1e-310, 50, 25}}) {
    auto table = sps_table();
    table["sample_steps_per_sec"] = invalid;
    EXPECT_FALSE(DSparkSpsCostModel::from_json(table.dump())) << table;
  }
}

TEST(DSparkSpsCostModelTest, RejectsInvalidBoundsAndOverflow) {
  for (const auto& invalid :
       nlohmann::json{0, -1, 7, 16.0, true, 4294967296ULL}) {
    auto table = sps_table();
    table["max_batch_tokens"] = invalid;
    EXPECT_FALSE(DSparkSpsCostModel::from_json(table.dump())) << table;
  }
  for (const auto& invalid : nlohmann::json{0, -1, 4096.0, true}) {
    auto table = additive_table();
    table["max_kv_length"] = invalid;
    EXPECT_FALSE(DSparkSpsCostModel::from_json(table.dump())) << table;
  }
  auto table = additive_table();
  table["bias_seconds"] = std::numeric_limits<double>::max();
  EXPECT_FALSE(DSparkSpsCostModel::from_json(table.dump()));
}

TEST(DSparkSpsCostModelTest, BothModelsDrivePhysicalGraphPlanner) {
  for (const auto& table : {sps_table(), additive_table()}) {
    const auto model = DSparkSpsCostModel::from_json(table.dump());
    ASSERT_TRUE(model);
    std::vector<DSparkGraphCost> costs;
    costs.reserve(3);
    for (const uint32_t graph : {2U, 4U, 8U}) {
      costs.emplace_back(
          DSparkGraphCost{graph, *model->step_time_ms(1, graph)});
    }
    DSparkBudgetOptions options;
    options.cost_scope = DSparkCostScope::FULL_STEP;
    const auto plan = DSparkBudgetPlanner::plan(
        1, std::vector<double>(7, 0.01), costs, options);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->graph_rows, 2U);
    EXPECT_EQ(plan->draft_budget, 1U);
  }
}

}  // namespace
}  // namespace xllm
