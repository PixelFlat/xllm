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

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "core/framework/speculative/adaptive/spec_cost_model.h"

namespace xllm {

// Copy this plan into a Task at admission; feedback must not alter queued
// Tasks. The Device allocator distributes exactly draft_budget proposals among
// the requests. Neither the allocation nor its per-request widths are read
// here.
struct VerifyBudgetPlan {
  uint32_t draft_budget = 0;
  uint32_t graph_rows = 0;
};

struct VerifyBudgetOptions {
  double fixed_draft_time_ms = 0.0;
  double commit_time_ms = 0.0;
  double min_gain = 0.0;
  uint32_t exploration_interval = 32;
  SpecCostScope cost_scope = SpecCostScope::VERIFY_TAIL;
  uint32_t draft_width = 7;
};

class VerifyBudgetPlanner final {
 public:
  // Dedicated calibration admission, independent of confidence/cost scoring.
  // A known G is used only within this physical draft bucket's warmed range.
  // Transient buckets outside that range explicitly run prewarmed full width;
  // their completions must not be recorded as samples of the requested G.
  // Unknown G, invalid geometry or missing full-width capacity is rejected.
  // Whole-round graphs may additionally restrict G for the current actual B.
  // Transient partial batches use their captured full width when necessary;
  // stable unsupported B/G pairs remain errors, not calibration samples.
  // Internal startup sampling supplies its intended B to identify transitions
  // even when the transient logical and physical request counts are equal.
  static std::optional<VerifyBudgetPlan> calibration_plan(
      uint32_t logical_requests,
      uint32_t physical_requests,
      uint32_t requested_graph_rows,
      std::span<const SpecGraphCost> prewarmed_graphs,
      uint32_t draft_width = 7,
      std::span<const SpecGraphCost> actual_batch_graphs = {},
      std::optional<uint32_t> calibration_requests = std::nullopt);

  // Lagged confidence is retired Host feedback, flattened [batch, draft_width],
  // or empty. Costs must be ascending, unique, prewarmed physical graph
  // buckets. Missing confidence or a usable full-width cost selects the warmed
  // full-width plan. A previous plan belongs to this batch geometry; request
  // identity/feedback generation checks and rounds_since_full_width are owned
  // by the caller. Invalid input or missing full-width graph capacity returns
  // std::nullopt before any execution state is mutated. This API never reads
  // Device tensors.
  static std::optional<VerifyBudgetPlan> plan(
      uint32_t batch_size,
      std::span<const double> lagged_conditional_confidence,
      std::span<const SpecGraphCost> prewarmed_costs,
      const VerifyBudgetOptions& options,
      std::optional<VerifyBudgetPlan> previous = std::nullopt,
      uint64_t rounds_since_full_width = 0);

  // Every shard receives the same retired feedback and computes the same
  // physical graph G. Budgets remain shard local; the score sums expected
  // progress across shards for the common measured step cost. A -1 feedback
  // value means an unseen request and requires that shard to run full width.
  // Empty shards keep budget zero but participate in the selected graph.
  static std::optional<std::vector<VerifyBudgetPlan>> parallel_plan(
      std::span<const int32_t> requests,
      std::span<const float> lagged_confidence,
      std::span<const uint64_t> rounds_since_full_width,
      std::span<const SpecGraphCost> prewarmed_costs,
      const VerifyBudgetOptions& options);
};

}  // namespace xllm
