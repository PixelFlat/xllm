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

#include "core/framework/speculative/dspark/dspark_window_collector.h"

#include <algorithm>
#include <utility>

namespace xllm {

DSparkSteadyWindowCollector::DSparkSteadyWindowCollector(
    DSparkWindowConfig config)
    : config_(std::move(config)), warmup_left_(config_.warmup_steps) {}

void DSparkSteadyWindowCollector::reset() {
  request_ids_.clear();
  window_ = {};
  last_generation_ = 0;
  last_ns_ = 0;
  window_start_ns_ = 0;
  warmup_left_ = config_.warmup_steps;
}

bool DSparkSteadyWindowCollector::finished() const {
  return emitted_ >= config_.max_windows;
}

std::optional<DSparkSteadyWindow> DSparkSteadyWindowCollector::observe(
    const DSparkSteadyStep& step) {
  if (finished()) {
    return std::nullopt;
  }
  if (step.generation == 0 || step.completed_ns == 0 ||
      step.logical_requests == 0 ||
      step.logical_requests != step.draft_requests ||
      step.request_ids.size() != step.logical_requests ||
      step.target_graph_rows != config_.target_graph_rows ||
      step.target_graph_rows < step.draft_requests ||
      step.max_kv_length < config_.min_kv_length ||
      step.max_kv_length > config_.max_kv_length ||
      step.draft_tokens !=
          std::min(static_cast<uint64_t>(step.logical_requests) * 7,
                   static_cast<uint64_t>(step.target_graph_rows -
                                         step.logical_requests)) ||
      step.accepted_tokens > step.draft_tokens) {
    reset();
    return std::nullopt;
  }
  if (std::any_of(step.request_ids.begin(),
                  step.request_ids.end(),
                  [](const std::string& id) { return id.empty(); })) {
    reset();
    return std::nullopt;
  }
  const bool same =
      last_generation_ != 0 && step.generation > last_generation_ &&
      step.generation - last_generation_ == 1 && step.completed_ns > last_ns_ &&
      window_.draft_requests == step.draft_requests &&
      std::equal(request_ids_.begin(),
                 request_ids_.end(),
                 step.request_ids.begin(),
                 step.request_ids.end());
  if (!same) {
    reset();
    request_ids_.assign(step.request_ids.begin(), step.request_ids.end());
    window_.logical_requests = step.logical_requests;
    window_.draft_requests = step.draft_requests;
    window_.target_graph_rows = step.target_graph_rows;
  }
  last_generation_ = step.generation;
  last_ns_ = step.completed_ns;
  if (!same || warmup_left_ != 0) {
    if (same) {
      --warmup_left_;
    }
    window_start_ns_ = step.completed_ns;
    window_.first_generation = step.generation;
    window_.min_kv_length = step.max_kv_length;
    window_.max_kv_length = step.max_kv_length;
    return std::nullopt;
  }
  window_.min_kv_length = std::min(window_.min_kv_length, step.max_kv_length);
  window_.max_kv_length = std::max(window_.max_kv_length, step.max_kv_length);
  ++window_.steps;
  window_.accepted_tokens += step.accepted_tokens;
  window_.draft_tokens += step.draft_tokens;
  if (window_.steps < config_.window_steps) {
    return std::nullopt;
  }
  window_.last_generation = step.generation;
  window_.elapsed_ms =
      static_cast<double>(step.completed_ns - window_start_ns_) / 1000000.0;
  const DSparkSteadyWindow result = window_;
  ++emitted_;
  window_start_ns_ = step.completed_ns;
  window_.first_generation = step.generation;
  window_.steps = 0;
  window_.accepted_tokens = 0;
  window_.draft_tokens = 0;
  window_.min_kv_length = step.max_kv_length;
  window_.max_kv_length = step.max_kv_length;
  return result;
}

}  // namespace xllm
