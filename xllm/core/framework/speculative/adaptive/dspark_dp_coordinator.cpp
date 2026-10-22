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

#include "core/framework/speculative/adaptive/dspark_dp_coordinator.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <utility>

namespace xllm {

DSparkDpCoordinator::DSparkDpCoordinator(uint32_t draft_width)
    : draft_width_(draft_width) {}

std::optional<DSparkDpFeedback> DSparkDpCoordinator::prepare(
    std::vector<std::vector<std::string>> sequence_ids,
    bool decode) {
  if ((draft_width_ != 5 && draft_width_ != 7) || sequence_ids.empty() ||
      pending_.size() >= 2) {
    return std::nullopt;
  }
  std::unordered_set<std::string> live;
  for (const auto& shard : sequence_ids) {
    for (const std::string& id : shard) {
      if (id.empty() || !live.insert(id).second) {
        return std::nullopt;
      }
    }
  }
  DSparkDpFeedback output;
  if (decode) {
    output.confidence.reserve(live.size() * draft_width_);
    output.rounds_since_full_width.reserve(live.size());
    for (const auto& shard : sequence_ids) {
      for (const std::string& id : shard) {
        const auto history = histories_.find(id);
        if (history == histories_.end()) {
          output.confidence.insert(
              output.confidence.end(), draft_width_, -1.0F);
          output.rounds_since_full_width.emplace_back(0);
          continue;
        }
        output.confidence.insert(
            output.confidence.end(),
            history->second.confidence.begin(),
            history->second.confidence.begin() + draft_width_);
        output.rounds_since_full_width.emplace_back(
            history->second.rounds_since_full_width);
      }
    }
  }
  for (const auto& pending : pending_) {
    for (const auto& shard : pending) {
      live.insert(shard.begin(), shard.end());
    }
  }
  std::erase_if(histories_,
                [&](const auto& entry) { return !live.contains(entry.first); });
  pending_.emplace_back(std::move(sequence_ids));
  return output;
}

bool DSparkDpCoordinator::retire(std::span<const DSparkDpFeedback> feedback) {
  if (pending_.empty() || feedback.size() != pending_.front().size()) {
    return false;
  }
  const auto& sequence_ids = pending_.front();
  for (uint32_t shard = 0; shard < feedback.size(); ++shard) {
    const auto& result = feedback[shard];
    if (result.confidence.empty() && result.rounds_since_full_width.empty()) {
      continue;
    }
    if (result.confidence.size() != sequence_ids[shard].size() * draft_width_ ||
        result.rounds_since_full_width.size() != sequence_ids[shard].size() ||
        std::any_of(result.confidence.begin(),
                    result.confidence.end(),
                    [](float value) {
                      return !std::isfinite(value) || value < 0.0F ||
                             value > 1.0F;
                    })) {
      return false;
    }
  }
  for (uint32_t shard = 0; shard < feedback.size(); ++shard) {
    const auto& result = feedback[shard];
    if (result.confidence.empty()) {
      continue;
    }
    for (uint32_t row = 0; row < sequence_ids[shard].size(); ++row) {
      Row history;
      std::copy_n(result.confidence.begin() + row * draft_width_,
                  draft_width_,
                  history.confidence.begin());
      history.rounds_since_full_width = result.rounds_since_full_width[row];
      histories_.insert_or_assign(sequence_ids[shard][row], std::move(history));
    }
  }
  pending_.pop_front();
  return true;
}

}  // namespace xllm
