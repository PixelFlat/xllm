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

#include "core/framework/speculative/adaptive/spec_cost_profile_store.h"

#include <gtest/gtest.h>

#include <limits>

namespace xllm {
namespace {

SpecCostProfile profile(uint32_t requests, uint32_t kv_bound) {
  return {requests,
          kv_bound,
          {{1, 1.0}, {4, 2.0}, {8, 3.0}, {16, 4.0}, {32, 5.0}},
          6.0,
          0.5};
}

TEST(SpecCostProfileStoreTest, MatchesSmallestCoveringKvBoundAndRequestBucket) {
  SpecCostProfileStore profiles;
  ASSERT_TRUE(profiles.store(profile(1, 4096)));
  ASSERT_TRUE(profiles.store(profile(1, 8192)));
  ASSERT_TRUE(profiles.store(profile(2, 4096)));
  ASSERT_TRUE(profiles.lookup(1, 2000).has_value());
  EXPECT_EQ(profiles.lookup(1, 2000)->max_kv_length, 4096U);
  EXPECT_EQ(profiles.lookup(1, 4096)->max_kv_length, 4096U);
  EXPECT_EQ(profiles.lookup(1, 4097)->max_kv_length, 8192U);
  EXPECT_EQ(profiles.lookup(2, 2000)->draft_requests, 2U);
  EXPECT_FALSE(profiles.lookup(2, 4097).has_value());
  EXPECT_FALSE(profiles.lookup(3, 2000).has_value());
  EXPECT_FALSE(profiles.lookup(1, 8193).has_value());
  EXPECT_FALSE(profiles.lookup(0, 2000).has_value());
  EXPECT_FALSE(profiles.lookup(1, 0).has_value());
}

TEST(SpecCostProfileStoreTest, SeparatesWidthsAtTheSameRequestBucket) {
  SpecCostProfileStore profiles;
  auto five = profile(5, 4096);
  five.draft_width = 5;
  five.target_graph_costs.emplace_back(SpecGraphCost{42, 6.0});
  ASSERT_TRUE(profiles.store(five));
  EXPECT_FALSE(profiles.lookup(5, 2000, 7));
  auto seven = profile(5, 4096);
  seven.target_graph_costs.emplace_back(SpecGraphCost{40, 7.0});
  ASSERT_TRUE(profiles.store(seven));
  ASSERT_TRUE(profiles.lookup(5, 2000, 5));
  ASSERT_TRUE(profiles.lookup(5, 2000, 7));
  EXPECT_EQ(profiles.lookup(5, 2000, 5)->draft_width, 5U);
  EXPECT_EQ(profiles.lookup(5, 2000, 7)->draft_width, 7U);
}

TEST(SpecCostProfileStoreTest, DraftFiveRequiresItsOwnFullWidthCapacity) {
  SpecCostProfileStore profiles;
  auto five = profile(2, 4096);
  five.draft_width = 5;
  five.target_graph_costs = {{1, 1.0}, {4, 2.0}, {8, 3.0}};
  EXPECT_FALSE(profiles.store(five));
  five.target_graph_costs.emplace_back(SpecGraphCost{12, 4.0});
  EXPECT_TRUE(profiles.store(five));
  five.draft_width = 0;
  EXPECT_FALSE(profiles.store(five));
}

TEST(SpecCostProfileStoreTest, DoesNotApplyContextTableAcrossCoverageGaps) {
  SpecCostProfileStore profiles;
  auto first = profile(1, 4096);
  first.min_kv_length = 1800;
  ASSERT_TRUE(profiles.store(first));
  auto second = profile(1, 8192);
  second.min_kv_length = 5000;
  ASSERT_TRUE(profiles.store(second));
  EXPECT_FALSE(profiles.lookup(1, 1799));
  ASSERT_TRUE(profiles.lookup(1, 1800));
  EXPECT_EQ(profiles.lookup(1, 4096)->max_kv_length, 4096U);
  EXPECT_FALSE(profiles.lookup(1, 4500));
  ASSERT_TRUE(profiles.lookup(1, 5000));
  EXPECT_EQ(profiles.lookup(1, 5000)->max_kv_length, 8192U);
  second.min_kv_length = 8193;
  EXPECT_FALSE(profiles.store(second));
  EXPECT_EQ(profiles.lookup(1, 5000)->min_kv_length, 5000U);
}

TEST(SpecCostProfileStoreTest, InvalidReplacementPreservesRetiredProfile) {
  SpecCostProfileStore profiles;
  auto initial = profile(1, 4096);
  ASSERT_TRUE(profiles.store(initial));
  initial.fixed_draft_time_ms = -1.0;
  EXPECT_FALSE(profiles.store(initial));
  ASSERT_TRUE(profiles.lookup(1, 100).has_value());
  EXPECT_EQ(profiles.lookup(1, 100)->fixed_draft_time_ms, 6.0);
  initial.fixed_draft_time_ms = 9.0;
  ASSERT_TRUE(profiles.store(std::move(initial)));
  EXPECT_EQ(profiles.lookup(1, 100)->fixed_draft_time_ms, 9.0);
}

TEST(SpecCostProfileStoreTest, RejectsMissingOrNonPositiveMeasurements) {
  SpecCostProfileStore profiles;
  auto invalid = profile(1, 4096);
  invalid.target_graph_costs.clear();
  EXPECT_FALSE(profiles.store(invalid));
  invalid = profile(1, 4096);
  invalid.fixed_draft_time_ms = 0.0;
  EXPECT_FALSE(profiles.store(invalid));
  invalid = profile(1, 4096);
  invalid.commit_time_ms = -1.0;
  EXPECT_FALSE(profiles.store(invalid));
  invalid = profile(1, 4096);
  invalid.target_graph_costs.front().target_time_ms = 0.0;
  EXPECT_FALSE(profiles.store(invalid));
  EXPECT_FALSE(profiles.lookup(1, 100).has_value());
}

TEST(SpecCostProfileStoreTest, RejectsNonFiniteTimingsAndOverflow) {
  SpecCostProfileStore profiles;
  auto invalid = profile(1, 4096);
  invalid.fixed_draft_time_ms = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(profiles.store(invalid));
  invalid = profile(1, 4096);
  invalid.commit_time_ms = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(profiles.store(invalid));
  invalid = profile(1, 4096);
  invalid.target_graph_costs.front().target_time_ms =
      std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(profiles.store(invalid));
  invalid = profile(1, 4096);
  invalid.fixed_draft_time_ms = std::numeric_limits<double>::max();
  invalid.target_graph_costs.back().target_time_ms =
      std::numeric_limits<double>::max();
  EXPECT_FALSE(profiles.store(invalid));
}

TEST(SpecCostProfileStoreTest, RequiresUniqueAscendingWarmedGraphGeometry) {
  SpecCostProfileStore profiles;
  auto invalid = profile(1, 4096);
  invalid.target_graph_costs[1].graph_rows = 0;
  EXPECT_FALSE(profiles.store(invalid));
  invalid = profile(1, 4096);
  invalid.target_graph_costs[1].graph_rows = 1;
  EXPECT_FALSE(profiles.store(invalid));
  invalid = profile(1, 4096);
  std::swap(invalid.target_graph_costs[0], invalid.target_graph_costs[1]);
  EXPECT_FALSE(profiles.store(invalid));
  invalid = profile(1, 4096);
  invalid.target_graph_costs.back().graph_rows =
      std::numeric_limits<uint32_t>::max();
  EXPECT_FALSE(profiles.store(invalid));
}

TEST(SpecCostProfileStoreTest, RequiresPositiveRequestsAndFullWidthCapacity) {
  SpecCostProfileStore profiles;
  EXPECT_FALSE(profiles.store(profile(0, 4096)));
  EXPECT_FALSE(profiles.store(profile(8, 4096)));
  EXPECT_FALSE(profiles.store(profile(1, 0)));
  EXPECT_TRUE(profiles.store(profile(4, 4096)));
}

TEST(SpecCostProfileStoreTest, DoesNotAssumeDraftTokenLayoutOrKernelCapacity) {
  SpecCostProfileStore profiles;
  for (const uint32_t width : {1U, 3U, 5U, 7U, 9U}) {
    for (const uint32_t requests : {1U, 3U, 129U}) {
      auto costs = profile(requests, /*kv_bound=*/4096);
      costs.draft_width = width;
      costs.target_graph_costs = {{requests * (width + 1), 1.0}};
      ASSERT_TRUE(profiles.store(std::move(costs)));
      const auto found =
          profiles.lookup(requests, /*max_kv_length=*/100, width);
      ASSERT_TRUE(found);
      EXPECT_EQ(found->draft_requests, requests);
      EXPECT_EQ(found->draft_width, width);
    }
  }
}

TEST(SpecCostProfileStoreTest, RejectsOverflowingFullWidthGeometry) {
  SpecCostProfileStore profiles;
  auto costs = profile(std::numeric_limits<uint32_t>::max(), /*kv_bound=*/4096);
  costs.target_graph_costs = {
      {static_cast<uint32_t>(std::numeric_limits<int32_t>::max()), 1.0}};
  EXPECT_FALSE(profiles.store(costs));
  costs.draft_requests = 1;
  costs.draft_width = std::numeric_limits<uint32_t>::max();
  EXPECT_FALSE(profiles.store(costs));
  costs.draft_requests = std::numeric_limits<uint32_t>::max();
  EXPECT_FALSE(profiles.store(costs));
  EXPECT_FALSE(profiles.lookup(/*draft_requests=*/1,
                               /*max_kv_length=*/100,
                               /*draft_width=*/0));
}

TEST(SpecCostProfileStoreTest, LookupReturnsIndependentImmutableSnapshot) {
  SpecCostProfileStore profiles;
  ASSERT_TRUE(profiles.store(profile(1, 4096)));
  auto task_profile = profiles.lookup(1, 100);
  ASSERT_TRUE(task_profile.has_value());
  task_profile->target_graph_costs[0].target_time_ms = 999.0;
  EXPECT_EQ(profiles.lookup(1, 100)->target_graph_costs[0].target_time_ms, 1.0);
  auto replacement = profile(1, 4096);
  replacement.fixed_draft_time_ms = 8.0;
  ASSERT_TRUE(profiles.store(std::move(replacement)));
  EXPECT_EQ(task_profile->fixed_draft_time_ms, 6.0);
  EXPECT_EQ(profiles.lookup(1, 100)->fixed_draft_time_ms, 8.0);
}

TEST(SpecCostProfileStoreTest, FullStepCostsCannotDoubleCountDraftOrCommit) {
  SpecCostProfileStore profiles;
  auto full = profile(1, 4096);
  full.cost_scope = SpecCostScope::FULL_STEP;
  EXPECT_FALSE(profiles.store(full));
  full.fixed_draft_time_ms = 0.0;
  full.commit_time_ms = 0.0;
  ASSERT_TRUE(profiles.store(full));
  EXPECT_EQ(profiles.lookup(1, 100)->cost_scope, SpecCostScope::FULL_STEP);
  full.commit_time_ms = 0.1;
  EXPECT_FALSE(profiles.store(full));
  EXPECT_EQ(profiles.lookup(1, 100)->commit_time_ms, 0.0);
}

}  // namespace
}  // namespace xllm
