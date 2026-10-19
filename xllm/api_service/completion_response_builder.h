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
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "api_service/utils.h"
#include "completion.pb.h"
#include "core/framework/request/request_output.h"

namespace xllm::api_service {

inline void set_logprobs(proto::Choice* choice,
                         const std::optional<std::vector<LogProb>>& logprobs,
                         int32_t offset = 0) {
  if (!logprobs.has_value() || logprobs.value().empty()) {
    return;
  }

  auto* proto_logprobs = choice->mutable_logprobs();
  // One entry per generated token, so the three parallel fields would otherwise
  // grow from empty and re-copy themselves O(log n) times per response.
  const int num_logprobs = static_cast<int>(logprobs.value().size());
  proto_logprobs->mutable_tokens()->Reserve(num_logprobs);
  proto_logprobs->mutable_token_ids()->Reserve(num_logprobs);
  proto_logprobs->mutable_token_logprobs()->Reserve(num_logprobs);
  for (const auto& logprob : logprobs.value()) {
    proto_logprobs->add_tokens(logprob.token);
    proto_logprobs->add_token_ids(logprob.token_id);
    proto_logprobs->add_token_logprobs(std::max(-9999.0f, logprob.logprob));
    proto_logprobs->add_text_offset(offset);
    offset += static_cast<int32_t>(std::count_if(
        logprob.token.begin(), logprob.token.end(), [](unsigned char byte) {
          return (byte & 0xc0) != 0x80;
        }));
    auto* top = proto_logprobs->add_top_logprobs()->mutable_values();
    if (logprob.top_logprobs.has_value()) {
      for (const auto& candidate : logprob.top_logprobs.value()) {
        (*top)[candidate.token] = std::max(-9999.0f, candidate.logprob);
      }
    }
  }
}

template <typename Emit>
bool build_completion_delta(proto::CompletionResponse& response,
                            bool include_usage,
                            const std::string& request_id,
                            int64_t created_time,
                            const std::string& model,
                            const RequestOutput& output,
                            std::unordered_map<size_t, int32_t>* offsets,
                            const Emit& emit) {
  for (const auto& seq_output : output.outputs) {
    if (!seq_output.text.empty()) {
      response.Clear();
      response.set_object("text_completion");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text(seq_output.text);
      int32_t& offset = (*offsets)[seq_output.index];
      set_logprobs(choice, seq_output.logprobs, offset);
      offset += static_cast<int32_t>(std::count_if(
          seq_output.text.begin(),
          seq_output.text.end(),
          [](unsigned char byte) { return (byte & 0xc0) != 0x80; }));
      if (!emit(response)) {
        return false;
      }
    }

    if (seq_output.finish_reason.has_value()) {
      response.Clear();
      response.set_object("text_completion");
      response.set_id(request_id);
      response.set_created(created_time);
      response.set_model(model);
      auto* choice = response.add_choices();
      choice->set_index(seq_output.index);
      choice->set_text("");
      choice->set_finish_reason(seq_output.finish_reason.value());
      api_service::set_proto_stop_reason(seq_output.stop_reason,
                                         choice->mutable_stop_reason());
      if (!emit(response)) {
        return false;
      }
    }
  }

  if (include_usage && output.usage.has_value() &&
      (output.finished || output.cancelled)) {
    const auto& usage = output.usage.value();
    response.Clear();
    response.set_object("text_completion");
    response.set_id(request_id);
    response.set_created(created_time);
    response.set_model(model);
    response.mutable_choices();
    auto* proto_usage = response.mutable_usage();
    api_service::set_proto_usage(proto_usage, usage);
    if (!emit(response)) {
      return false;
    }
  }

  return true;
}

inline void build_completion_result(proto::CompletionResponse& response,
                                    const std::string& request_id,
                                    int64_t created_time,
                                    const std::string& model,
                                    const RequestOutput& req_output) {
  response.Clear();
  response.set_object("text_completion");
  response.set_id(request_id);
  response.set_created(created_time);
  response.set_model(model);

  response.mutable_choices()->Reserve(req_output.outputs.size());
  for (const auto& output : req_output.outputs) {
    auto* choice = response.add_choices();
    choice->set_index(output.index);
    api_service::set_proto_stop_reason(output.stop_reason,
                                       choice->mutable_stop_reason());
    choice->set_text(output.text);
    set_logprobs(choice, output.logprobs);
    if (output.finish_reason.has_value()) {
      choice->set_finish_reason(output.finish_reason.value());
    }
  }

  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto* proto_usage = response.mutable_usage();
    api_service::set_proto_usage(proto_usage, usage);
  }
}

}  // namespace xllm::api_service
