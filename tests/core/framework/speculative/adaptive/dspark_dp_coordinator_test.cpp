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

#include "core/framework/speculative/adaptive/dspark_dp_coordinator.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <vector>

namespace xllm {
namespace {

TEST(DSparkDpCoordinatorTest, UsesOnlyRetiredFeedbackAcrossOverlappedTasks) {
  DSparkDpCoordinator coordinator(7);
  const auto first = coordinator.prepare({{"a"}, {"b"}}, true);
  ASSERT_TRUE(first);
  EXPECT_EQ(first->confidence, std::vector<float>(14, -1.0F));
  const auto second = coordinator.prepare({{"a"}, {"b"}}, true);
  ASSERT_TRUE(second);
  EXPECT_EQ(second->confidence, first->confidence);
  EXPECT_FALSE(coordinator.prepare({{"a"}, {"b"}}, true));
  const std::array<DSparkDpFeedback, 2> feedback = {
      DSparkDpFeedback{std::vector<float>(7, 0.2F), {3}},
      DSparkDpFeedback{std::vector<float>(7, 0.8F), {5}}};
  ASSERT_TRUE(coordinator.retire(feedback));
  const auto third = coordinator.prepare({{"b", "a"}, {"new"}}, true);
  ASSERT_TRUE(third);
  std::vector<float> expected(7, 0.8F);
  expected.insert(expected.end(), 7, 0.2F);
  expected.insert(expected.end(), 7, -1.0F);
  EXPECT_EQ(third->confidence, expected);
  EXPECT_EQ(third->rounds_since_full_width, (std::vector<uint64_t>{5, 3, 0}));
}

TEST(DSparkDpCoordinatorTest, RejectsMalformedRetirementWithoutConsumingTask) {
  DSparkDpCoordinator coordinator(5);
  ASSERT_TRUE(coordinator.prepare({{}, {"a"}}, true));
  std::array<DSparkDpFeedback, 2> feedback = {DSparkDpFeedback{},
                                              DSparkDpFeedback{{0.5F}, {0}}};
  EXPECT_FALSE(coordinator.retire(feedback));
  feedback[1].confidence.assign(5, 0.5F);
  feedback[1].confidence[0] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_FALSE(coordinator.retire(feedback));
  feedback[1].confidence[0] = 0.5F;
  ASSERT_TRUE(coordinator.retire(feedback));
  const auto next = coordinator.prepare({{}, {"a"}}, true);
  ASSERT_TRUE(next);
  EXPECT_EQ(next->confidence, std::vector<float>(5, 0.5F));
}

TEST(DSparkDpCoordinatorTest, PrefillAndFinishedRequestsDoNotLeakHistory) {
  DSparkDpCoordinator coordinator(7);
  ASSERT_TRUE(coordinator.prepare({{"old"}, {}}, false));
  const std::array<DSparkDpFeedback, 2> empty{};
  ASSERT_TRUE(coordinator.retire(empty));
  ASSERT_TRUE(coordinator.prepare({{"old"}, {}}, true));
  const std::array<DSparkDpFeedback, 2> feedback = {
      DSparkDpFeedback{std::vector<float>(7, 0.9F), {1}}, DSparkDpFeedback{}};
  ASSERT_TRUE(coordinator.retire(feedback));
  ASSERT_TRUE(coordinator.prepare({{"new"}, {}}, false));
  ASSERT_TRUE(coordinator.retire(empty));
  const auto reused = coordinator.prepare({{"old"}, {}}, true);
  ASSERT_TRUE(reused);
  EXPECT_EQ(reused->confidence, std::vector<float>(7, -1.0F));
}

TEST(DSparkDpCoordinatorTest, RejectsInvalidWidthAndAmbiguousIdentity) {
  DSparkDpCoordinator invalid(6);
  EXPECT_FALSE(invalid.prepare({{"a"}, {}}, true));
  DSparkDpCoordinator coordinator(7);
  EXPECT_FALSE(coordinator.prepare({{"a"}, {"a"}}, true));
  EXPECT_FALSE(coordinator.prepare({{""}, {}}, true));
  EXPECT_FALSE(coordinator.prepare({}, true));
  EXPECT_TRUE(coordinator.prepare({{"a"}, {"b"}}, true));
}

}  // namespace
}  // namespace xllm
