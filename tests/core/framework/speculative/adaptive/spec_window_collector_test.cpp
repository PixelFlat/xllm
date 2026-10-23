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

#include "core/framework/speculative/adaptive/spec_window_collector.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "core/framework/speculative/adaptive/verify_budget_planner.h"

namespace xllm {
namespace {

SpecWindowConfig make_config() {
  SpecWindowConfig config;
  config.target_graph_rows = 8;
  config.max_kv_length = 4096;
  config.warmup_steps = 0;
  config.window_steps = 2;
  return config;
}

SpecSteadyStep make_step(uint64_t generation,
                         uint64_t completed_ns,
                         const std::vector<std::string>& ids) {
  SpecSteadyStep step;
  step.generation = generation;
  step.completed_ns = completed_ns;
  step.logical_requests = static_cast<uint32_t>(ids.size());
  step.draft_requests = step.logical_requests;
  step.target_graph_rows = 8;
  step.max_kv_length = 2000;
  step.accepted_tokens = 2;
  step.draft_tokens = 7;
  step.request_ids = ids;
  return step;
}

TEST(SpecWindowCollectorTest, MeasuresIntervalsNotCompletionCount) {
  SpecSteadyWindowCollector collector(make_config());
  const std::vector<std::string> ids = {"a"};
  EXPECT_FALSE(collector.observe(make_step(1, 1000000, ids)));
  EXPECT_FALSE(collector.observe(make_step(2, 21000000, ids)));
  const auto window = collector.observe(make_step(3, 41000000, ids));
  ASSERT_TRUE(window);
  EXPECT_EQ(window->steps, 2U);
  EXPECT_EQ(window->first_generation, 1U);
  EXPECT_EQ(window->last_generation, 3U);
  EXPECT_DOUBLE_EQ(window->elapsed_ms, 40.0);
  EXPECT_EQ(window->accepted_tokens, 4U);
  EXPECT_EQ(window->draft_tokens, 14U);
}

TEST(SpecWindowCollectorTest, RecordsConfiguredDraftFiveWidth) {
  auto config = make_config();
  config.draft_width = 5;
  SpecSteadyWindowCollector collector(config);
  const std::vector<std::string> ids = {"draft-five"};
  for (uint64_t generation = 1; generation <= 3; ++generation) {
    auto step = make_step(generation, generation * 1000000, ids);
    step.draft_tokens = 5;
    const auto window = collector.observe(step);
    if (generation == 3) {
      ASSERT_TRUE(window);
      EXPECT_EQ(window->draft_tokens, 10U);
    } else {
      EXPECT_FALSE(window);
    }
  }
}

TEST(SpecWindowCollectorTest, AveragesAcrossBurstRetirementGaps) {
  auto config = make_config();
  config.window_steps = 4;
  SpecSteadyWindowCollector collector(config);
  const std::vector<std::string> ids = {"a"};
  EXPECT_FALSE(collector.observe(make_step(1, 1, ids)));
  EXPECT_FALSE(collector.observe(make_step(2, 40000001, ids)));
  EXPECT_FALSE(collector.observe(make_step(3, 40000002, ids)));
  EXPECT_FALSE(collector.observe(make_step(4, 80000000, ids)));
  const auto window = collector.observe(make_step(5, 80000001, ids));
  ASSERT_TRUE(window);
  EXPECT_DOUBLE_EQ(window->elapsed_ms / window->steps, 20.0);
}

TEST(SpecWindowCollectorTest, RecordsMultiRequestProgressWithoutBonusTokens) {
  auto config = make_config();
  config.target_graph_rows = 64;
  SpecSteadyWindowCollector collector(config);
  std::vector<std::string> ids;
  ids.reserve(8);
  for (uint32_t row = 0; row < 8; ++row) {
    ids.emplace_back(std::to_string(row));
  }
  std::optional<SpecSteadyWindow> window;
  for (uint64_t generation = 1; generation <= 3; ++generation) {
    auto step = make_step(generation, generation * 20000000, ids);
    step.target_graph_rows = 64;
    step.accepted_tokens = 16;
    step.draft_tokens = 56;
    window = collector.observe(step);
  }
  ASSERT_TRUE(window);
  EXPECT_EQ(window->logical_requests, 8U);
  EXPECT_EQ(window->draft_requests, 8U);
  EXPECT_EQ(window->accepted_tokens, 32U);
  EXPECT_EQ(window->draft_tokens, 112U);
  EXPECT_DOUBLE_EQ(window->elapsed_ms, 40.0);
}

TEST(SpecWindowCollectorTest, DiscardsSettlingIntervalsAndRestartsAfterReset) {
  auto config = make_config();
  config.warmup_steps = 2;
  SpecSteadyWindowCollector collector(config);
  const std::vector<std::string> ids = {"a"};
  for (uint64_t generation = 1; generation <= 4; ++generation) {
    EXPECT_FALSE(
        collector.observe(make_step(generation, generation * 10, ids)));
  }
  const auto window = collector.observe(make_step(5, 50, ids));
  ASSERT_TRUE(window);
  EXPECT_EQ(window->first_generation, 3U);
  EXPECT_EQ(window->steps, 2U);
  collector.reset();
  EXPECT_FALSE(collector.observe(make_step(6, 60, ids)));
  EXPECT_FALSE(collector.observe(make_step(7, 70, ids)));
}

TEST(SpecWindowCollectorTest, ResetsOnTaskGapsAndNonmonotonicClock) {
  for (const bool clock_reset : {false, true}) {
    SpecSteadyWindowCollector collector(make_config());
    const std::vector<std::string> ids = {"a"};
    EXPECT_FALSE(collector.observe(make_step(1, 100, ids)));
    EXPECT_FALSE(collector.observe(make_step(2, 200, ids)));
    EXPECT_FALSE(collector.observe(make_step(clock_reset ? 3 : 4, 50, ids)));
    EXPECT_FALSE(collector.observe(make_step(clock_reset ? 4 : 5, 300, ids)));
    const auto window =
        collector.observe(make_step(clock_reset ? 5 : 6, 400, ids));
    ASSERT_TRUE(window);
    EXPECT_DOUBLE_EQ(window->elapsed_ms, 0.00035);
  }
}

TEST(SpecWindowCollectorTest, ResetsOnMembershipChange) {
  SpecSteadyWindowCollector collector(make_config());
  const std::vector<std::string> first = {"a"};
  const std::vector<std::string> second = {"b"};
  EXPECT_FALSE(collector.observe(make_step(1, 10, first)));
  EXPECT_FALSE(collector.observe(make_step(2, 20, first)));
  EXPECT_FALSE(collector.observe(make_step(3, 30, second)));
  EXPECT_FALSE(collector.observe(make_step(4, 40, second)));
  const auto window = collector.observe(make_step(5, 50, second));
  ASSERT_TRUE(window);
  EXPECT_EQ(window->first_generation, 3U);
}

TEST(SpecWindowCollectorTest, RejectsPrefillPartialBucketsAndInvalidCounters) {
  const std::vector<std::string> ids = {"a"};
  for (uint32_t kind = 0; kind < 7; ++kind) {
    SpecSteadyWindowCollector collector(make_config());
    EXPECT_FALSE(collector.observe(make_step(1, 10, ids)));
    EXPECT_FALSE(collector.observe(make_step(2, 20, ids)));
    auto step = make_step(3, 30, ids);
    switch (kind) {
      case 0:
        step.logical_requests = 0;
        break;
      case 1:
        step.draft_requests = 2;
        break;
      case 2:
        step.target_graph_rows = 4;
        break;
      case 3:
        step.max_kv_length = 4097;
        break;
      case 4:
        step.draft_tokens = 8;
        break;
      case 5:
        step.accepted_tokens = 8;
        break;
      case 6:
        step.request_ids = {};
        break;
    }
    EXPECT_FALSE(collector.observe(step));
    EXPECT_FALSE(collector.observe(make_step(4, 40, ids)));
    EXPECT_FALSE(collector.observe(make_step(5, 50, ids)));
    ASSERT_TRUE(collector.observe(make_step(6, 60, ids)));
  }
}

TEST(SpecWindowCollectorTest, ReportsKvRangeAndCapsWindowCount) {
  auto config = make_config();
  config.max_windows = 1;
  SpecSteadyWindowCollector collector(config);
  const std::vector<std::string> ids = {"a"};
  EXPECT_FALSE(collector.observe(make_step(1, 10, ids)));
  auto step = make_step(2, 20, ids);
  step.max_kv_length = 2010;
  EXPECT_FALSE(collector.observe(step));
  step = make_step(3, 30, ids);
  step.max_kv_length = 2020;
  const auto window = collector.observe(step);
  ASSERT_TRUE(window);
  EXPECT_EQ(window->min_kv_length, 2000U);
  EXPECT_EQ(window->max_kv_length, 2020U);
  EXPECT_TRUE(collector.finished());
  collector.reset();
  EXPECT_TRUE(collector.finished());
  EXPECT_FALSE(collector.observe(make_step(4, 40, ids)));
}

TEST(SpecWindowCollectorTest,
     ExcludesTransientTasksOutsideCalibratedGraphRange) {
  auto config = make_config();
  config.target_graph_rows = 64;
  SpecSteadyWindowCollector collector(config);
  const std::vector<SpecGraphCost> graphs = {{1, 0.0},
                                             {2, 0.0},
                                             {4, 0.0},
                                             {8, 0.0},
                                             {16, 0.0},
                                             {32, 0.0},
                                             {48, 0.0},
                                             {64, 0.0}};
  const std::vector<std::string> single = {"transient"};
  const std::vector<std::string> batch = {
      "a", "b", "c", "d", "e", "f", "g", "h"};
  auto step = make_step(1, 10, single);
  const auto transient =
      VerifyBudgetPlanner::calibration_plan(/*logical_requests=*/1,
                                            /*physical_requests=*/1,
                                            /*requested_graph_rows=*/64,
                                            graphs);
  ASSERT_TRUE(transient);
  step.target_graph_rows = transient->graph_rows;
  step.draft_tokens = transient->draft_budget;
  EXPECT_FALSE(collector.observe(step));
  std::optional<SpecSteadyWindow> window;
  const auto steady = VerifyBudgetPlanner::calibration_plan(
      /*logical_requests=*/8,
      /*physical_requests=*/8,
      /*requested_graph_rows=*/64,
      graphs);
  ASSERT_TRUE(steady);
  for (uint64_t generation = 2; generation <= 4; ++generation) {
    step = make_step(generation, generation * 10, batch);
    step.target_graph_rows = steady->graph_rows;
    step.draft_tokens = steady->draft_budget;
    window = collector.observe(step);
  }
  ASSERT_TRUE(window);
  EXPECT_EQ(window->first_generation, 2U);
  EXPECT_EQ(window->last_generation, 4U);
  EXPECT_EQ(window->draft_requests, 8U);
  EXPECT_EQ(window->target_graph_rows, 64U);
  EXPECT_DOUBLE_EQ(window->elapsed_ms, 0.00002);
}

}  // namespace
}  // namespace xllm
