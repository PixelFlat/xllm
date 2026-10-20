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

#include "core/framework/speculative/dspark/dspark_adaptive_cost_cache.h"

#include <gtest/gtest.h>

#include <limits>

namespace xllm {
namespace {

DSparkAdaptiveCostProfile profile(uint32_t rows, uint32_t kv_bound) {
  return {rows,
          kv_bound,
          {{1, 1.0}, {4, 2.0}, {8, 3.0}, {16, 4.0}, {32, 5.0}},
          6.0,
          0.5};
}

TEST(DSparkAdaptiveCostCacheTest, MatchesSmallestCoveringKvBoundAndDraftGraph) {
  DSparkAdaptiveCostCache cache;
  ASSERT_TRUE(cache.store(profile(7, 4096)));
  ASSERT_TRUE(cache.store(profile(7, 8192)));
  ASSERT_TRUE(cache.store(profile(14, 4096)));
  ASSERT_TRUE(cache.lookup(7, 2000).has_value());
  EXPECT_EQ(cache.lookup(7, 2000)->max_kv_length, 4096U);
  EXPECT_EQ(cache.lookup(7, 4096)->max_kv_length, 4096U);
  EXPECT_EQ(cache.lookup(7, 4097)->max_kv_length, 8192U);
  EXPECT_EQ(cache.lookup(14, 2000)->draft_graph_rows, 14U);
  EXPECT_FALSE(cache.lookup(14, 4097).has_value());
  EXPECT_FALSE(cache.lookup(21, 2000).has_value());
  EXPECT_FALSE(cache.lookup(7, 8193).has_value());
  EXPECT_FALSE(cache.lookup(0, 2000).has_value());
  EXPECT_FALSE(cache.lookup(7, 0).has_value());
}

TEST(DSparkAdaptiveCostCacheTest, DoesNotApplyContextTableAcrossCoverageGaps) {
  DSparkAdaptiveCostCache cache;
  auto first = profile(7, 4096);
  first.min_kv_length = 1800;
  ASSERT_TRUE(cache.store(first));
  auto second = profile(7, 8192);
  second.min_kv_length = 5000;
  ASSERT_TRUE(cache.store(second));
  EXPECT_FALSE(cache.lookup(7, 1799));
  ASSERT_TRUE(cache.lookup(7, 1800));
  EXPECT_EQ(cache.lookup(7, 4096)->max_kv_length, 4096U);
  EXPECT_FALSE(cache.lookup(7, 4500));
  ASSERT_TRUE(cache.lookup(7, 5000));
  EXPECT_EQ(cache.lookup(7, 5000)->max_kv_length, 8192U);
  second.min_kv_length = 8193;
  EXPECT_FALSE(cache.store(second));
  EXPECT_EQ(cache.lookup(7, 5000)->min_kv_length, 5000U);
}

TEST(DSparkAdaptiveCostCacheTest, InvalidReplacementPreservesRetiredProfile) {
  DSparkAdaptiveCostCache cache;
  auto initial = profile(7, 4096);
  ASSERT_TRUE(cache.store(initial));
  initial.fixed_draft_time_ms = -1.0;
  EXPECT_FALSE(cache.store(initial));
  ASSERT_TRUE(cache.lookup(7, 100).has_value());
  EXPECT_EQ(cache.lookup(7, 100)->fixed_draft_time_ms, 6.0);
  initial.fixed_draft_time_ms = 9.0;
  ASSERT_TRUE(cache.store(std::move(initial)));
  EXPECT_EQ(cache.lookup(7, 100)->fixed_draft_time_ms, 9.0);
}

TEST(DSparkAdaptiveCostCacheTest, RejectsMissingOrNonPositiveMeasurements) {
  DSparkAdaptiveCostCache cache;
  auto invalid = profile(7, 4096);
  invalid.target_graph_costs.clear();
  EXPECT_FALSE(cache.store(invalid));
  invalid = profile(7, 4096);
  invalid.fixed_draft_time_ms = 0.0;
  EXPECT_FALSE(cache.store(invalid));
  invalid = profile(7, 4096);
  invalid.commit_time_ms = -1.0;
  EXPECT_FALSE(cache.store(invalid));
  invalid = profile(7, 4096);
  invalid.target_graph_costs.front().target_time_ms = 0.0;
  EXPECT_FALSE(cache.store(invalid));
  EXPECT_FALSE(cache.lookup(7, 100).has_value());
}

TEST(DSparkAdaptiveCostCacheTest, RejectsNonFiniteTimingsAndOverflow) {
  DSparkAdaptiveCostCache cache;
  auto invalid = profile(7, 4096);
  invalid.fixed_draft_time_ms = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(cache.store(invalid));
  invalid = profile(7, 4096);
  invalid.commit_time_ms = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(cache.store(invalid));
  invalid = profile(7, 4096);
  invalid.target_graph_costs.front().target_time_ms =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(cache.store(invalid));
  invalid = profile(7, 4096);
  invalid.fixed_draft_time_ms = std::numeric_limits<double>::max();
  invalid.target_graph_costs.back().target_time_ms =
      std::numeric_limits<double>::max();
  EXPECT_FALSE(cache.store(invalid));
}

TEST(DSparkAdaptiveCostCacheTest, RequiresUniqueAscendingWarmedGraphGeometry) {
  DSparkAdaptiveCostCache cache;
  auto invalid = profile(7, 4096);
  invalid.target_graph_costs[1].graph_rows = 0;
  EXPECT_FALSE(cache.store(invalid));
  invalid = profile(7, 4096);
  invalid.target_graph_costs[1].graph_rows = 1;
  EXPECT_FALSE(cache.store(invalid));
  invalid = profile(7, 4096);
  std::swap(invalid.target_graph_costs[0], invalid.target_graph_costs[1]);
  EXPECT_FALSE(cache.store(invalid));
  invalid = profile(7, 4096);
  invalid.target_graph_costs.back().graph_rows =
      std::numeric_limits<uint32_t>::max();
  EXPECT_FALSE(cache.store(invalid));
}

TEST(DSparkAdaptiveCostCacheTest,
     RequiresFixedDraft7AndCompleteFullWidthCapacity) {
  DSparkAdaptiveCostCache cache;
  EXPECT_FALSE(cache.store(profile(0, 4096)));
  EXPECT_FALSE(cache.store(profile(8, 4096)));
  EXPECT_FALSE(cache.store(profile(7 * 129, 4096)));
  EXPECT_FALSE(cache.store(profile(7, 0)));
  EXPECT_FALSE(cache.store(profile(7 * 8, 4096)));
  EXPECT_TRUE(cache.store(profile(7 * 4, 4096)));
}

TEST(DSparkAdaptiveCostCacheTest, LookupReturnsIndependentImmutableSnapshot) {
  DSparkAdaptiveCostCache cache;
  ASSERT_TRUE(cache.store(profile(7, 4096)));
  auto task_profile = cache.lookup(7, 100);
  ASSERT_TRUE(task_profile.has_value());
  task_profile->target_graph_costs[0].target_time_ms = 999.0;
  EXPECT_EQ(cache.lookup(7, 100)->target_graph_costs[0].target_time_ms, 1.0);
  auto replacement = profile(7, 4096);
  replacement.fixed_draft_time_ms = 8.0;
  ASSERT_TRUE(cache.store(std::move(replacement)));
  EXPECT_EQ(task_profile->fixed_draft_time_ms, 6.0);
  EXPECT_EQ(cache.lookup(7, 100)->fixed_draft_time_ms, 8.0);
}

TEST(DSparkAdaptiveCostCacheTest, FullStepCostsCannotDoubleCountDraftOrCommit) {
  DSparkAdaptiveCostCache cache;
  auto full = profile(7, 4096);
  full.cost_scope = DSparkCostScope::FULL_STEP;
  EXPECT_FALSE(cache.store(full));
  full.fixed_draft_time_ms = 0.0;
  full.commit_time_ms = 0.0;
  ASSERT_TRUE(cache.store(full));
  EXPECT_EQ(cache.lookup(7, 100)->cost_scope, DSparkCostScope::FULL_STEP);
  full.commit_time_ms = 0.1;
  EXPECT_FALSE(cache.store(full));
  EXPECT_EQ(cache.lookup(7, 100)->commit_time_ms, 0.0);
}

}  // namespace
}  // namespace xllm
