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

#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "core/framework/speculative/draft_step_selector.h"

namespace xllm {

struct DraftStepCalibrationPoint {
  uint32_t draft_steps = 0;
  DSparkCalibrationPoint batch;
};

// Engine-owned draft-step calibration. ProfileManager supplies real requests;
// completed ordinary result fences feed the shared DSpark steady-window
// collector. No timing Events, extra Device readback, or offline cost JSON.
class DraftStepStartupCalibration final {
 public:
  DraftStepStartupCalibration(std::string configuration_path,
                              uint32_t max_steps);
  const DSparkStartupCalibrationConfig& configuration(uint32_t max_positions);
  void set_geometry(std::vector<uint32_t> requests,
                    std::vector<uint32_t> graphs);
  std::vector<DraftStepCalibrationPoint> points() const;
  void begin_sample(DraftStepCalibrationPoint point);
  void dispatched(std::span<const std::string> request_ids);
  void observe(std::span<const DraftStepAcceptanceObservation> observations,
               uint64_t completed_ns);
  std::vector<DSparkSteadyWindow> finish_sample();
  void install_costs(std::vector<std::vector<DSparkCostTableEntry>> costs);
  bool sampling() const { return point_.has_value(); }
  uint32_t forced_steps() const { return point_ ? point_->draft_steps : 0; }
  uint32_t forced_graph_rows() const {
    return point_ ? point_->batch.target_graph_rows : 0;
  }
  uint32_t select(uint32_t requests, uint32_t max_kv_length);

 private:
  struct Pending {
    uint64_t generation = 0;
    uint64_t statistics_epoch = 0;
    std::vector<std::string> request_ids;
  };
  std::string configuration_path_;
  const uint32_t max_steps_;
  DraftStepSelectorConfig policy_;
  std::optional<DSparkStartupCalibrationConfig> configuration_;
  std::vector<uint32_t> captured_requests_;
  std::vector<uint32_t> captured_graphs_;
  std::optional<DraftStepCalibrationPoint> point_;
  std::deque<Pending> pending_;
  uint64_t generation_ = 0;
  std::vector<uint32_t> kv_lengths_;
  std::unique_ptr<DSparkSteadyWindowCollector> collector_;
  std::vector<DSparkSteadyWindow> windows_;
  std::unique_ptr<DraftStepSelector> selector_;
};

}  // namespace xllm
