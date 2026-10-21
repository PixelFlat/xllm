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

#include <glog/logging.h>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <vector>

#include "core/runtime/task_execution_pipeline_speculative.h"

namespace xllm {

std::optional<std::string> TaskExecutionPipeline::dspark_calibration(
    DSparkCalibrationCommand command) {
  check_external_thread();
  folly::Promise<std::optional<std::string>> promise;
  auto future = promise.getFuture();
  state_executor_.schedule([this,
                            command = std::move(command),
                            promise = std::move(promise)]() mutable {
    if (!speculative_capacity_ || !speculative_capacity_->startup_calibration ||
        !dspark_adaptive_ || dspark_adaptive_->startup_calibration_complete ||
        !accepted_.empty()) {
      promise.setValue(std::nullopt);
      return;
    }
    CHECK(completed_.empty());
    promise.setValue(dspark_calibration_impl(command));
  });
  return std::move(future).get();
}

std::optional<std::string> TaskExecutionPipeline::dspark_calibration_impl(
    const DSparkCalibrationCommand& command) {
  std::vector<uint32_t> requests;
  std::vector<uint32_t> graphs;
  const uint32_t draft_width = speculative_capacity_->num_speculative_tokens;
  const auto& first = *slots_.front()->speculative->drafts.front();
  for (const int64_t rows : first.captured_graph_batch_sizes) {
    if (rows <= 0 || rows % draft_width != 0) {
      return std::nullopt;
    }
    const bool common =
        std::all_of(slots_.begin(), slots_.end(), [&](const auto& slot) {
          const auto& captured =
              slot->speculative->drafts.front()->captured_graph_batch_sizes;
          return std::binary_search(captured.begin(), captured.end(), rows);
        });
    if (common) {
      requests.emplace_back(static_cast<uint32_t>(rows / draft_width));
    }
  }
  graphs.reserve(captured_graph_batch_sizes_.size());
  for (const int64_t rows : captured_graph_batch_sizes_) {
    graphs.emplace_back(static_cast<uint32_t>(rows));
  }
  const auto graph_coverage = adaptive_dspark_graph_coverage();
  const auto kv_bound = [&](uint32_t batch) {
    uint64_t bound =
        std::min(capacity_.max_kv_seq_len, capacity_.max_positions);
    for (const auto& slot : slots_) {
      const auto& draft = *slot->speculative->drafts.front();
      const auto captured =
          draft.captured_graph_kv_capacities.find(batch * draft_width);
      if (captured == draft.captured_graph_kv_capacities.end()) {
        return uint32_t{0};
      }
      bound = std::min(
          {bound,
           captured->second + 1,
           static_cast<uint64_t>(draft.input->model_params()
                                     .attention.device.block_tables.size(1)) *
               capacity_.logical_block_size});
    }
    return static_cast<uint32_t>(bound);
  };
  switch (command.action) {
    case DSparkCalibrationAction::CAPABILITIES: {
      nlohmann::json bounds = nlohmann::json::array();
      for (const uint32_t batch : requests) {
        bounds.emplace_back(nlohmann::json{{"requests", batch},
                                           {"max_kv_length", kv_bound(batch)}});
      }
      nlohmann::json coverage = nlohmann::json::array();
      for (const auto& entry : graph_coverage) {
        coverage.emplace_back(
            nlohmann::json{{"requests", entry.requests},
                           {"target_graph_rows", entry.target_graph_rows}});
      }
      return nlohmann::json{{"request_buckets", requests},
                            {"draft_width", draft_width},
                            {"target_graph_rows", graphs},
                            {"graph_coverage", std::move(coverage)},
                            {"kv_bounds", std::move(bounds)},
                            {"slot_count", capacity_.slot_count}}
          .dump();
    }
    case DSparkCalibrationAction::BEGIN_SAMPLE: {
      if (dspark_adaptive_->calibration_point) {
        return std::nullopt;
      }
      const auto data = nlohmann::json::parse(command.payload, nullptr, false);
      if (!data.is_object() || !data.contains("configuration") ||
          !data.contains("point") || !data["point"].is_number_unsigned()) {
        return std::nullopt;
      }
      const auto config = parse_dspark_startup_calibration(
          data["configuration"].dump(), draft_width);
      if (!config) {
        return std::nullopt;
      }
      const auto points =
          plan_dspark_calibration(*config, requests, graphs, graph_coverage);
      const uint64_t index = data["point"].get<uint64_t>();
      if (!points || index >= points->size()) {
        return std::nullopt;
      }
      const auto& point = (*points)[index];
      if (point.context.max_kv_length > kv_bound(point.requests)) {
        return std::nullopt;
      }
      dspark_adaptive_->calibration_point = point;
      if (speculative_capacity_->calibration_leader) {
        dspark_adaptive_->calibration_session =
            std::make_unique<DSparkCalibrationSession>(*config, point);
      }
      return std::string("sample_started");
    }
    case DSparkCalibrationAction::FINISH_SAMPLE: {
      if (!dspark_adaptive_->calibration_point) {
        return std::nullopt;
      }
      nlohmann::json result = nlohmann::json::array();
      const auto& session = dspark_adaptive_->calibration_session;
      if (speculative_capacity_->calibration_leader) {
        if (!session || !session->finished()) {
          return std::nullopt;
        }
        for (const auto& window : session->windows()) {
          result.emplace_back(
              nlohmann::json{{"first_generation", window.first_generation},
                             {"last_generation", window.last_generation},
                             {"logical_requests", window.logical_requests},
                             {"draft_requests", window.draft_requests},
                             {"target_graph_rows", window.target_graph_rows},
                             {"min_kv_length", window.min_kv_length},
                             {"max_kv_length", window.max_kv_length},
                             {"steps", window.steps},
                             {"accepted_tokens", window.accepted_tokens},
                             {"draft_tokens", window.draft_tokens},
                             {"elapsed_ms", window.elapsed_ms}});
        }
      }
      dspark_adaptive_->calibration_session.reset();
      dspark_adaptive_->calibration_point.reset();
      return result.dump();
    }
    case DSparkCalibrationAction::INSTALL_COSTS: {
      if (dspark_adaptive_->calibration_point) {
        return std::nullopt;
      }
      const auto bundle = DSparkCostTableBundle::from_json(command.payload);
      if (!bundle) {
        return std::nullopt;
      }
      auto costs = std::make_shared<DSparkAdaptiveCostCache>();
      size_t installed_profiles = 0;
      for (const auto& entry : bundle->entries()) {
        const auto& covered_requests =
            entry.request_buckets.empty() ? requests : entry.request_buckets;
        for (const uint32_t batch : covered_requests) {
          const auto full = std::lower_bound(
              graphs.begin(), graphs.end(), batch * (draft_width + 1));
          if (!std::binary_search(requests.begin(), requests.end(), batch) ||
              full == graphs.end() ||
              (!entry.request_buckets.empty() &&
               entry.max_kv_length > kv_bound(batch))) {
            return std::nullopt;
          }
          if (entry.request_buckets.empty() &&
              !entry.model.step_time_ms(batch, *full)) {
            continue;
          }
          DSparkAdaptiveCostProfile profile;
          profile.draft_width = draft_width;
          profile.draft_graph_rows = batch * draft_width;
          profile.min_kv_length = entry.min_kv_length;
          profile.max_kv_length =
              std::min(entry.max_kv_length, kv_bound(batch));
          profile.cost_scope = DSparkCostScope::FULL_STEP;
          for (const uint32_t rows : graphs) {
            if (rows < batch || rows > *full) {
              continue;
            }
            const auto cost = entry.model.step_time_ms(batch, rows);
            if (!cost) {
              return std::nullopt;
            }
            profile.target_graph_costs.emplace_back(
                DSparkGraphCost{rows, *cost});
          }
          if (!costs->store(std::move(profile))) {
            return std::nullopt;
          }
          ++installed_profiles;
        }
      }
      if (installed_profiles == 0) {
        return std::nullopt;
      }
      dspark_adaptive_->owned_costs = std::move(costs);
      speculative_capacity_->adaptive_costs = dspark_adaptive_->owned_costs;
      dspark_adaptive_->histories.assign(dspark_adaptive_->histories.size(),
                                         {});
      dspark_adaptive_->retired_plans = {};
      std::fill(context_->published_.begin(), context_->published_.end(), 0);
      context_->request_ids_.assign(context_->request_ids_.size(), {});
      dspark_adaptive_->startup_calibration_complete = true;
      // Echo the canonical table, allowing the master to confirm every rank
      // installed exactly the same immutable costs before declaring ready.
      return command.payload;
    }
  }
  return std::nullopt;
}

}  // namespace xllm
