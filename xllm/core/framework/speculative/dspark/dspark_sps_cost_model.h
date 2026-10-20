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
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace xllm {

// Immutable, Host-only DSpark step costs. JSON fields and units match SGLang's
// SPS and additive tables. Evaluate physical graph rows, not logical B+K:
// padding does not reduce the replay's cost. The additive B axis represents
// the physical Draft request bucket (Bcap), including padded requests.
class DSparkSpsCostModel final {
 public:
  enum class Kind : uint8_t { SPS, ADDITIVE };

  // Reject malformed, ambiguous, non-finite or non-positive cost models.
  // Optional max_kv_length bounds the contexts for which a table is valid.
  static std::optional<DSparkSpsCostModel> from_json(std::string_view text);

  Kind kind() const { return kind_; }
  uint32_t max_kv_length() const { return max_kv_length_; }

  std::optional<double> step_time_ms(uint32_t draft_requests,
                                     uint32_t target_graph_rows) const;

 private:
  DSparkSpsCostModel() = default;

  Kind kind_ = Kind::SPS;
  uint32_t max_batch_tokens_ = std::numeric_limits<uint32_t>::max();
  uint32_t max_kv_length_ = std::numeric_limits<uint32_t>::max();
  double bias_seconds_ = 0.0;
  std::vector<uint32_t> bs_probes_;
  std::vector<double> alpha_seconds_;
  std::vector<uint32_t> token_probes_;
  // SPS: full step milliseconds. Additive: theta seconds.
  std::vector<double> token_values_;
};

}  // namespace xllm
