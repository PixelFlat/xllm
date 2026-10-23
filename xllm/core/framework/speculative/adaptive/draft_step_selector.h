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
#include <deque>
#include <span>
#include <string>
#include <vector>

#include "core/framework/speculative/adaptive/spec_startup_calibration.h"

namespace xllm {

struct DraftStepSelectorConfig {
  uint32_t max_steps = 0;
  uint32_t window_size = 20;
  uint32_t update_interval = 5;
  double switch_min_gain = 0.05;
  // One SPS/additive full-cycle model per candidate depth, measured for the
  // current draft/target pair. Reuse the shared cost-table representation,
  // bounds and interpolation, not another strategy's measurements.
  std::vector<std::vector<SpecCostTableEntry>> costs;
};

struct DraftStepAcceptanceObservation {
  uint32_t draft_steps = 0;
  uint32_t accepted_drafts = 0;
};

DraftStepSelectorConfig load_draft_step_selector_config(const std::string& path,
                                                        uint32_t max_steps);

// Engine-owned Host-only policy. Observe completed result metadata, then stamp
// one immutable decision into the next task for every TP rank. No current-step
// confidence, device readback, process group, or global worker option is used.
class DraftStepSelector final {
 public:
  explicit DraftStepSelector(DraftStepSelectorConfig config);

  // One span is one retired verify batch, regardless of its request count.
  // The dispatch epoch prevents older in-flight batches from repopulating
  // deeper windows after a depth decrease.
  void observe(std::span<const DraftStepAcceptanceObservation> observations,
               uint64_t statistics_epoch);
  uint32_t select(uint32_t batch_size,
                  uint32_t max_kv_length,
                  uint32_t draft_requests,
                  std::span<const uint32_t> target_graph_rows);
  std::vector<double> position_acceptance() const;
  uint64_t statistics_epoch() const { return statistics_epoch_; }

 private:
  void clear_positions_above(uint32_t steps);

  DraftStepSelectorConfig config_;
  std::vector<std::deque<double>> histories_;
  std::vector<double> acceptance_sums_;
  std::vector<uint64_t> minimum_epochs_;
  uint32_t current_steps_ = 0;
  uint64_t statistics_epoch_ = 0;
  uint64_t completed_batches_ = 0;
  uint64_t last_update_ = 0;
};

}  // namespace xllm
