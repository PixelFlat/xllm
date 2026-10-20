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

#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/speculative/dspark/dspark_budget_planner.h"

namespace xllm {

struct DSparkAdaptiveCostProfile {
  uint32_t draft_graph_rows = 0;
  // Conservative attention-length bound of the warmed input binding.
  uint32_t max_kv_length = 0;
  std::vector<DSparkGraphCost> target_graph_costs;
  double fixed_draft_time_ms = 0.0;
  double commit_time_ms = 0.0;
  DSparkCostScope cost_scope = DSparkCostScope::VERIFY_TAIL;
  // Contexts outside this calibrated interval must not reuse its costs.
  uint32_t min_kv_length = 1;
};

// State-thread-owned, startup-only profile construction. Store profiles only
// after all ranks agree on calibrated costs; serving lookups never read Device
// data or modify the profile of an already admitted Task.
class DSparkAdaptiveCostCache final {
 public:
  // Reject malformed/incomplete measurements without changing the cache.
  bool store(DSparkAdaptiveCostProfile profile);
  // Match the smallest warmed KV bound covering this prepared input. Missing
  // capacity/profile is an explicit full-width admission policy, not an
  // execution retry. Both graph rows and KV geometry must be covered.
  std::optional<DSparkAdaptiveCostProfile> lookup(uint32_t draft_graph_rows,
                                                  uint32_t max_kv_length) const;

 private:
  std::map<std::pair<uint32_t, uint32_t>, DSparkAdaptiveCostProfile> profiles_;
};

}  // namespace xllm
