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
#include <string>
#include <vector>

namespace xllm {

struct DSparkWindowConfig {
  uint32_t target_graph_rows = 0;
  uint32_t min_kv_length = 1;
  uint32_t max_kv_length = 0;
  uint32_t warmup_steps = 8;
  uint32_t window_steps = 32;
  uint32_t max_windows = 3;
  uint32_t draft_width = 7;
};

// Borrowed Host snapshots of a retired Task; no Device reads or timing Events.
struct DSparkSteadyStep {
  uint64_t generation = 0;
  uint64_t completed_ns = 0;
  uint32_t logical_requests = 0;
  uint32_t draft_requests = 0;
  uint32_t target_graph_rows = 0;
  uint32_t max_kv_length = 0;
  uint32_t accepted_tokens = 0;
  uint32_t draft_tokens = 0;
  std::span<const std::string> request_ids;
};

struct DSparkSteadyWindow {
  uint64_t first_generation = 0;
  uint64_t last_generation = 0;
  uint32_t logical_requests = 0;
  uint32_t draft_requests = 0;
  uint32_t target_graph_rows = 0;
  uint32_t min_kv_length = 0;
  uint32_t max_kv_length = 0;
  uint32_t steps = 0;
  uint64_t accepted_tokens = 0;
  uint64_t draft_tokens = 0;
  double elapsed_ms = 0.0;
};

// Startup-only in-memory collection. No files, writer thread or serving mode.
class DSparkSteadyWindowCollector final {
 public:
  explicit DSparkSteadyWindowCollector(DSparkWindowConfig config);
  // Gaps, prefill, membership/shape changes and KV gaps reset the window.
  std::optional<DSparkSteadyWindow> observe(const DSparkSteadyStep& step);
  void reset();
  bool finished() const;

 private:
  DSparkWindowConfig config_;
  std::vector<std::string> request_ids_;
  DSparkSteadyWindow window_;
  uint64_t last_generation_ = 0;
  uint64_t last_ns_ = 0;
  uint64_t window_start_ns_ = 0;
  uint32_t warmup_left_ = 0;
  uint32_t emitted_ = 0;
};

}  // namespace xllm
