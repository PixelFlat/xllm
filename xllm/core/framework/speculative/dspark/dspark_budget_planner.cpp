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

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

namespace xllm {
namespace {

bool valid_options(const DSparkBudgetOptions& options) {
  if (options.draft_width != 5 && options.draft_width != 7) {
    return false;
  }
  if (options.cost_scope != DSparkCostScope::VERIFY_TAIL &&
      options.cost_scope != DSparkCostScope::FULL_STEP) {
    return false;
  }
  if (options.cost_scope == DSparkCostScope::FULL_STEP &&
      (options.fixed_draft_time_ms != 0.0 || options.commit_time_ms != 0.0)) {
    return false;
  }
  return std::isfinite(options.fixed_draft_time_ms) &&
         options.fixed_draft_time_ms >= 0.0 &&
         std::isfinite(options.commit_time_ms) &&
         options.commit_time_ms >= 0.0 &&
         std::isfinite(options.fixed_draft_time_ms + options.commit_time_ms) &&
         std::isfinite(options.min_gain) && options.min_gain >= 0.0 &&
         options.exploration_interval != 0;
}

bool valid_costs(std::span<const DSparkGraphCost> costs, double fixed_time_ms) {
  return !costs.empty() &&
         std::all_of(
             costs.begin(),
             costs.end(),
             [&](const auto& cost) {
               return cost.graph_rows != 0 &&
                      std::isfinite(cost.target_time_ms) &&
                      cost.target_time_ms >= 0.0 &&
                      std::isfinite(fixed_time_ms + cost.target_time_ms);
             }) &&
         std::adjacent_find(
             costs.begin(), costs.end(), [](const auto& lhs, const auto& rhs) {
               return lhs.graph_rows >= rhs.graph_rows;
             }) == costs.end();
}

}  // namespace

std::optional<DSparkBudgetPlan> DSparkBudgetPlanner::calibration_plan(
    uint32_t logical_requests,
    uint32_t physical_requests,
    uint32_t requested_graph_rows,
    std::span<const DSparkGraphCost> prewarmed_graphs,
    uint32_t draft_width,
    std::span<const DSparkGraphCost> actual_batch_graphs,
    std::optional<uint32_t> calibration_requests) {
  if ((draft_width != 5 && draft_width != 7) || logical_requests == 0 ||
      logical_requests > physical_requests ||
      physical_requests >
          std::numeric_limits<uint32_t>::max() / (draft_width + 1) ||
      !valid_costs(prewarmed_graphs, /*fixed_time_ms=*/0.0) ||
      (!actual_batch_graphs.empty() &&
       !valid_costs(actual_batch_graphs, /*fixed_time_ms=*/0.0)) ||
      (calibration_requests && *calibration_requests == 0)) {
    return std::nullopt;
  }
  const auto find_rows = [&](uint32_t rows) {
    return std::lower_bound(
        prewarmed_graphs.begin(),
        prewarmed_graphs.end(),
        rows,
        [](const DSparkGraphCost& graph, uint32_t requested) {
          return graph.graph_rows < requested;
        });
  };
  const auto requested = find_rows(requested_graph_rows);
  const auto full = find_rows(physical_requests * (draft_width + 1));
  if (requested == prewarmed_graphs.end() ||
      requested->graph_rows != requested_graph_rows ||
      full == prewarmed_graphs.end()) {
    return std::nullopt;
  }
  if (std::any_of(actual_batch_graphs.begin(),
                  actual_batch_graphs.end(),
                  [&](const DSparkGraphCost& graph) {
                    const auto known = find_rows(graph.graph_rows);
                    return graph.graph_rows < logical_requests ||
                           graph.graph_rows > full->graph_rows ||
                           known == prewarmed_graphs.end() ||
                           known->graph_rows != graph.graph_rows;
                  })) {
    return std::nullopt;
  }
  // Request admission/retirement can temporarily select a smaller or larger
  // Bcap than the calibration workload. Decide the supported shape before
  // executing a Task; never capture a new binding or retry an execution.
  uint32_t graph_rows = requested_graph_rows >= physical_requests &&
                                requested_graph_rows <= full->graph_rows
                            ? requested_graph_rows
                            : full->graph_rows;
  if (!actual_batch_graphs.empty()) {
    const auto actual =
        std::lower_bound(actual_batch_graphs.begin(),
                         actual_batch_graphs.end(),
                         graph_rows,
                         [](const DSparkGraphCost& graph, uint32_t rows) {
                           return graph.graph_rows < rows;
                         });
    if (actual == actual_batch_graphs.end() ||
        actual->graph_rows != graph_rows) {
      // A partially admitted/retired batch can share Bcap without sharing the
      // whole-round B/G binding. Use its warmed full width before execution;
      // never substitute a different tier for a stable requested sample.
      if (logical_requests == physical_requests &&
          requested_graph_rows >= physical_requests &&
          requested_graph_rows <= full->graph_rows &&
          (!calibration_requests ||
           *calibration_requests == logical_requests)) {
        return std::nullopt;
      }
      const auto actual_full =
          std::lower_bound(actual_batch_graphs.begin(),
                           actual_batch_graphs.end(),
                           logical_requests * (draft_width + 1),
                           [](const DSparkGraphCost& graph, uint32_t rows) {
                             return graph.graph_rows < rows;
                           });
      if (actual_full == actual_batch_graphs.end()) {
        return std::nullopt;
      }
      graph_rows = actual_full->graph_rows;
    }
  }
  return DSparkBudgetPlan{
      std::min(logical_requests * draft_width, graph_rows - logical_requests),
      graph_rows};
}

std::optional<DSparkBudgetPlan> DSparkBudgetPlanner::plan(
    uint32_t batch_size,
    std::span<const double> lagged_conditional_confidence,
    std::span<const DSparkGraphCost> prewarmed_costs,
    const DSparkBudgetOptions& options,
    std::optional<DSparkBudgetPlan> previous,
    uint64_t rounds_since_full_width) {
  const uint32_t draft_width = options.draft_width;
  if (batch_size == 0 || !valid_options(options) ||
      batch_size > std::numeric_limits<uint32_t>::max() / (draft_width + 1)) {
    return std::nullopt;
  }
  const double fixed_time_ms =
      options.fixed_draft_time_ms + options.commit_time_ms;
  if (!valid_costs(prewarmed_costs, fixed_time_ms)) {
    return std::nullopt;
  }
  const uint32_t max_budget = batch_size * draft_width;
  const uint32_t full_rows = batch_size + max_budget;
  if ((!lagged_conditional_confidence.empty() &&
       lagged_conditional_confidence.size() != max_budget) ||
      std::any_of(lagged_conditional_confidence.begin(),
                  lagged_conditional_confidence.end(),
                  [](double confidence) {
                    return !std::isfinite(confidence) || confidence < 0.0 ||
                           confidence > 1.0;
                  })) {
    return std::nullopt;
  }
  const auto full_cost =
      std::lower_bound(prewarmed_costs.begin(),
                       prewarmed_costs.end(),
                       full_rows,
                       [](const DSparkGraphCost& cost, uint32_t rows) {
                         return cost.graph_rows < rows;
                       });
  if (full_cost == prewarmed_costs.end()) {
    return std::nullopt;
  }
  const DSparkBudgetPlan full_plan{max_budget, full_cost->graph_rows};

  const auto previous_cost =
      previous.has_value()
          ? std::lower_bound(prewarmed_costs.begin(),
                             prewarmed_costs.end(),
                             previous->graph_rows,
                             [](const DSparkGraphCost& cost, uint32_t rows) {
                               return cost.graph_rows < rows;
                             })
          : prewarmed_costs.end();
  if (previous.has_value() &&
      (previous->draft_budget > max_budget ||
       static_cast<uint64_t>(batch_size) + previous->draft_budget >
           previous->graph_rows ||
       previous_cost == prewarmed_costs.end() ||
       previous_cost->graph_rows != previous->graph_rows)) {
    return std::nullopt;
  }
  // This explicit admission policy is not an execution fallback: no Task has
  // been accepted yet, and only a supplied, prewarmed graph can be selected.
  if (lagged_conditional_confidence.empty() ||
      (options.cost_scope == DSparkCostScope::VERIFY_TAIL &&
       options.fixed_draft_time_ms == 0.0) ||
      full_cost->target_time_ms == 0.0 ||
      rounds_since_full_width >= options.exploration_interval) {
    return full_plan;
  }

  std::vector<double> path_probabilities;
  path_probabilities.reserve(max_budget);
  for (uint32_t row = 0; row < batch_size; ++row) {
    double path_probability = 1.0;
    for (uint32_t step = 0; step < draft_width; ++step) {
      path_probability *=
          lagged_conditional_confidence[row * draft_width + step];
      path_probabilities.emplace_back(path_probability);
    }
  }
  std::sort(path_probabilities.begin(),
            path_probabilities.end(),
            std::greater<double>());
  // Conditional probabilities produce non-increasing paths in each request.
  // The top K values therefore describe a feasible prefix allocation; ties
  // can always be assigned to earlier positions in that request.
  std::vector<double> expected_progress;
  expected_progress.reserve(static_cast<size_t>(max_budget) + 1);
  expected_progress.emplace_back(static_cast<double>(batch_size));
  for (double probability : path_probabilities) {
    expected_progress.emplace_back(expected_progress.back() + probability);
  }
  const auto score = [&](const DSparkBudgetPlan& candidate,
                         const DSparkGraphCost& cost) {
    return expected_progress[candidate.draft_budget] /
           (fixed_time_ms + cost.target_time_ms);
  };

  DSparkBudgetPlan best = full_plan;
  double best_score = -1.0;
  for (const DSparkGraphCost& cost : prewarmed_costs) {
    if (cost.graph_rows < batch_size || cost.target_time_ms == 0.0) {
      continue;
    }
    // Within the same physical graph, additional prefix rows have no target
    // cost. Spend all available capacity, bounded by the configured draft
    // block.
    const DSparkBudgetPlan candidate{
        std::min(max_budget, cost.graph_rows - batch_size), cost.graph_rows};
    const double candidate_score = score(candidate, cost);
    if (candidate_score > best_score) {
      best = candidate;
      best_score = candidate_score;
    }
  }
  if (previous.has_value() && previous_cost->target_time_ms > 0.0) {
    const DSparkBudgetPlan previous_filled{
        std::min(max_budget, previous->graph_rows - batch_size),
        previous->graph_rows};
    const double previous_score = score(previous_filled, *previous_cost);
    if (best_score <= previous_score * (1.0 + options.min_gain)) {
      return previous_filled;
    }
  }
  return best;
}

}  // namespace xllm
