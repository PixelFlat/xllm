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

namespace xllm {

enum class DSparkCostScope : uint8_t { VERIFY_TAIL, FULL_STEP };

struct DSparkGraphCost {
  uint32_t graph_rows = 0;
  // VERIFY_TAIL or FULL_STEP, as selected by DSparkBudgetOptions::cost_scope.
  // Zero denotes a warmed graph without a usable latency measurement.
  double target_time_ms = 0.0;
};

// Copy this plan into a Task at admission; feedback must not alter queued
// Tasks. The Device allocator distributes exactly draft_budget proposals among
// the requests. Neither the allocation nor its per-request widths are read
// here.
struct DSparkBudgetPlan {
  uint32_t draft_budget = 0;
  uint32_t graph_rows = 0;
};

struct DSparkBudgetOptions {
  double fixed_draft_time_ms = 0.0;
  double commit_time_ms = 0.0;
  double min_gain = 0.0;
  uint32_t exploration_interval = 32;
  DSparkCostScope cost_scope = DSparkCostScope::VERIFY_TAIL;
};

class DSparkBudgetPlanner final {
 public:
  // Dedicated calibration admission, independent of confidence/cost scoring.
  // A known G is used only within this physical draft bucket's warmed range.
  // Transient buckets outside that range explicitly run prewarmed full width;
  // their completions must not be recorded as samples of the requested G.
  // Unknown G, invalid geometry or missing full-width capacity is rejected.
  static std::optional<DSparkBudgetPlan> calibration_plan(
      uint32_t logical_requests,
      uint32_t physical_requests,
      uint32_t requested_graph_rows,
      std::span<const DSparkGraphCost> prewarmed_graphs);

  // Lagged confidence is retired Host feedback, flattened [batch, 7], or
  // empty. Costs must be ascending, unique, prewarmed physical graph buckets.
  // Missing confidence or a usable full-width cost selects the warmed
  // full-width plan. A previous plan belongs to this batch geometry; request
  // identity/feedback generation checks and rounds_since_full_width are owned
  // by the caller. Invalid input or missing full-width graph capacity returns
  // std::nullopt before any execution state is mutated. This API never reads
  // Device tensors.
  static std::optional<DSparkBudgetPlan> plan(
      uint32_t batch_size,
      std::span<const double> lagged_conditional_confidence,
      std::span<const DSparkGraphCost> prewarmed_costs,
      const DSparkBudgetOptions& options,
      std::optional<DSparkBudgetPlan> previous = std::nullopt,
      uint64_t rounds_since_full_width = 0);
};

}  // namespace xllm
