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

#include <algorithm>
#include <cmath>
#include <limits>

namespace xllm {

bool DSparkAdaptiveCostCache::store(DSparkAdaptiveCostProfile profile) {
  const uint32_t draft_width = profile.draft_width;
  const bool valid_scope =
      (profile.cost_scope == DSparkCostScope::VERIFY_TAIL &&
       profile.fixed_draft_time_ms > 0.0) ||
      (profile.cost_scope == DSparkCostScope::FULL_STEP &&
       profile.fixed_draft_time_ms == 0.0 && profile.commit_time_ms == 0.0);
  if ((draft_width != 5 && draft_width != 7) || profile.draft_graph_rows == 0 ||
      profile.draft_graph_rows % draft_width != 0 ||
      profile.draft_graph_rows / draft_width > 128 ||
      profile.min_kv_length == 0 ||
      profile.max_kv_length < profile.min_kv_length ||
      !std::isfinite(profile.fixed_draft_time_ms) || !valid_scope ||
      !std::isfinite(profile.commit_time_ms) || profile.commit_time_ms < 0.0 ||
      !std::isfinite(profile.fixed_draft_time_ms + profile.commit_time_ms) ||
      profile.target_graph_costs.empty()) {
    return false;
  }
  const auto& costs = profile.target_graph_costs;
  if (std::any_of(
          costs.begin(),
          costs.end(),
          [&](const DSparkGraphCost& cost) {
            return cost.graph_rows == 0 ||
                   cost.graph_rows > std::numeric_limits<int32_t>::max() ||
                   !std::isfinite(cost.target_time_ms) ||
                   cost.target_time_ms <= 0.0 ||
                   !std::isfinite(profile.fixed_draft_time_ms +
                                  profile.commit_time_ms + cost.target_time_ms);
          }) ||
      std::adjacent_find(
          costs.begin(),
          costs.end(),
          [](const DSparkGraphCost& lhs, const DSparkGraphCost& rhs) {
            return lhs.graph_rows >= rhs.graph_rows;
          }) != costs.end() ||
      costs.back().graph_rows <
          profile.draft_graph_rows / draft_width * (draft_width + 1)) {
    return false;
  }
  const auto key = std::make_tuple(
      draft_width, profile.draft_graph_rows, profile.max_kv_length);
  profiles_.insert_or_assign(key, std::move(profile));
  return true;
}

std::optional<DSparkAdaptiveCostProfile> DSparkAdaptiveCostCache::lookup(
    uint32_t draft_graph_rows,
    uint32_t max_kv_length,
    uint32_t draft_width) const {
  if (draft_graph_rows == 0 || max_kv_length == 0) {
    return std::nullopt;
  }
  const auto found =
      profiles_.lower_bound({draft_width, draft_graph_rows, max_kv_length});
  if (found == profiles_.end() || found->second.draft_width != draft_width ||
      found->second.draft_graph_rows != draft_graph_rows ||
      found->second.min_kv_length > max_kv_length) {
    return std::nullopt;
  }
  return found->second;
}

}  // namespace xllm
