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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {

// Host-only feedback already covered by the ordinary result-readback fence.
// Rows are request ordered, and each confidence row has draft_width columns.
struct DSparkDpFeedback {
  std::vector<float> confidence;
  std::vector<uint64_t> rounds_since_full_width;
};

inline bool valid_dspark_dp_feedback(const DSparkDpFeedback& feedback,
                                     uint64_t rows) {
  if (feedback.confidence.empty() && feedback.rounds_since_full_width.empty()) {
    return true;
  }
  return rows != 0 && feedback.rounds_since_full_width.size() == rows &&
         (feedback.confidence.size() == rows * 5 ||
          feedback.confidence.size() == rows * 7) &&
         std::all_of(feedback.confidence.begin(),
                     feedback.confidence.end(),
                     [](float value) {
                       return std::isfinite(value) && value >= 0.0F &&
                              value <= 1.0F;
                     });
}

// The engine retires n-2 before dispatching n. Replicate that retired feedback
// through the existing input transport instead of synchronizing devices between
// Draft and Target. Identity snapshots keep completed/reordered requests from
// lending confidence to a different request or newly reused embedding slot.
class DSparkDpCoordinator final {
 public:
  explicit DSparkDpCoordinator(uint32_t draft_width);

  std::optional<DSparkDpFeedback> prepare(
      std::vector<std::vector<std::string>> sequence_ids,
      bool decode);
  bool retire(std::span<const DSparkDpFeedback> feedback);

 private:
  struct Row {
    std::array<float, 7> confidence{};
    uint64_t rounds_since_full_width = 0;
  };

  uint32_t draft_width_;
  std::deque<std::vector<std::vector<std::string>>> pending_;
  std::unordered_map<std::string, Row> histories_;
};

}  // namespace xllm
