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

#include "core/framework/speculative/draft_step_startup_calibration.h"

#include <glog/logging.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <utility>

namespace xllm {

DraftStepStartupCalibration::DraftStepStartupCalibration(
    std::string configuration_path,
    uint32_t max_steps)
    : configuration_path_(std::move(configuration_path)),
      max_steps_(max_steps),
      policy_(load_draft_step_selector_config(configuration_path_, max_steps)) {
  CHECK_GE(max_steps, 1U);
  CHECK_LE(max_steps, 64U);
}

const DSparkStartupCalibrationConfig&
DraftStepStartupCalibration::configuration(uint32_t max_positions) {
  if (configuration_) {
    return *configuration_;
  }
  if (configuration_path_.empty()) {
    configuration_ = default_dspark_startup_calibration(
        max_positions, policy_.max_steps, /*autoregressive_draft=*/true);
  } else {
    std::ifstream stream(configuration_path_);
    CHECK(stream.is_open());
    const auto json = nlohmann::json::parse(stream);
    CHECK(json.is_object());
    configuration_ =
        json.contains("calibration")
            ? parse_dspark_startup_calibration(json.at("calibration").dump(),
                                               max_steps_,
                                               /*autoregressive_draft=*/true)
            : default_dspark_startup_calibration(max_positions,
                                                 max_steps_,
                                                 /*autoregressive_draft=*/true);
  }
  CHECK(configuration_) << "Invalid draft-step calibration settings; provide "
                           "sampling settings with sufficient KV headroom.";
  LOG(INFO) << "Draft-step startup calibration effective configuration: source="
            << (configuration_path_.empty() ? "built-in defaults"
                                            : configuration_path_)
            << ", calibration="
            << serialize_dspark_startup_calibration(*configuration_)
            << ", policy="
            << nlohmann::json{{"window_size", policy_.window_size},
                              {"update_interval", policy_.update_interval},
                              {"switch_min_gain", policy_.switch_min_gain}}
                   .dump();
  return *configuration_;
}

void DraftStepStartupCalibration::set_geometry(std::vector<uint32_t> requests,
                                               std::vector<uint32_t> graphs) {
  CHECK(configuration_ && !sampling() && !selector_);
  CHECK(!requests.empty() && !graphs.empty());
  CHECK(std::is_sorted(requests.begin(), requests.end()));
  CHECK(std::is_sorted(graphs.begin(), graphs.end()));
  captured_requests_ = std::move(requests);
  captured_graphs_ = std::move(graphs);
  if (configuration_->request_buckets.empty()) {
    configuration_->request_buckets = captured_requests_;
  }
  for (const uint32_t batch : configuration_->request_buckets) {
    CHECK_LE(batch, 128U);
    CHECK(std::binary_search(
        captured_requests_.begin(), captured_requests_.end(), batch));
    CHECK_GE(captured_graphs_.back(),
             static_cast<uint64_t>(batch) * (policy_.max_steps + 1));
  }
  for (const uint32_t rows : configuration_->target_graph_rows) {
    CHECK(std::binary_search(
        captured_graphs_.begin(), captured_graphs_.end(), rows));
  }
  const auto& sampled_graphs = configuration_->target_graph_rows.empty()
                                   ? captured_graphs_
                                   : configuration_->target_graph_rows;
  CHECK_GE(sampled_graphs.back(),
           static_cast<uint64_t>(configuration_->request_buckets.back()) *
               (max_steps_ + 1));
}

std::vector<DraftStepCalibrationPoint> DraftStepStartupCalibration::points()
    const {
  CHECK(configuration_ && !captured_requests_.empty());
  const auto& config = *configuration_;
  const auto& graphs = config.target_graph_rows.empty()
                           ? captured_graphs_
                           : config.target_graph_rows;
  std::vector<DraftStepCalibrationPoint> points;
  points.reserve(config.contexts.size() * policy_.max_steps *
                 config.request_buckets.size() * graphs.size());
  for (const auto& context : config.contexts) {
    for (uint32_t depth = 1; depth <= policy_.max_steps; ++depth) {
      for (const uint32_t requests : config.request_buckets) {
        const uint64_t logical_rows =
            static_cast<uint64_t>(requests) * (depth + 1);
        // Include all legal padded G values so the shared additive fit can
        // distinguish the B axis from the G axis instead of fitting a rank-
        // deficient diagonal of minimum-verify shapes.
        for (const uint32_t rows : graphs) {
          if (rows < logical_rows) {
            continue;
          }
          points.emplace_back(DraftStepCalibrationPoint{
              depth, DSparkCalibrationPoint{requests, rows, context}});
        }
      }
    }
  }
  return points;
}

void DraftStepStartupCalibration::begin_sample(
    DraftStepCalibrationPoint point) {
  CHECK(configuration_ && !sampling() && !selector_ && pending_.empty());
  CHECK_GE(point.draft_steps, 1U);
  CHECK_LE(point.draft_steps, policy_.max_steps);
  CHECK_GT(point.batch.requests, 0U);
  CHECK_GE(
      point.batch.target_graph_rows,
      static_cast<uint64_t>(point.batch.requests) * (point.draft_steps + 1));
  point_ = point;
  kv_lengths_.assign(point.batch.requests,
                     point.batch.context.prefill_tokens + 1);
  DSparkWindowConfig config;
  config.target_graph_rows = point.batch.target_graph_rows;
  config.min_kv_length = point.batch.context.min_kv_length;
  config.max_kv_length = point.batch.context.max_kv_length;
  config.warmup_steps = configuration_->warmup_steps;
  config.window_steps = configuration_->window_steps;
  config.max_windows = configuration_->windows_per_point;
  config.draft_width = point.draft_steps;
  config.autoregressive_draft = true;
  collector_ = std::make_unique<DSparkSteadyWindowCollector>(config);
  windows_.clear();
  windows_.reserve(config.max_windows);
}

void DraftStepStartupCalibration::dispatched(
    std::span<const std::string> request_ids) {
  CHECK(point_ || selector_);
  Pending pending;
  if (point_) {
    CHECK_EQ(request_ids.size(), point_->batch.requests);
    pending.generation = ++generation_;
    pending.request_ids.assign(request_ids.begin(), request_ids.end());
  } else {
    pending.statistics_epoch = selector_->statistics_epoch();
  }
  pending_.emplace_back(std::move(pending));
}

void DraftStepStartupCalibration::observe(
    std::span<const DraftStepAcceptanceObservation> observations,
    uint64_t completed_ns) {
  if (!sampling()) {
    if (selector_) {
      CHECK(!pending_.empty());
      selector_->observe(observations, pending_.front().statistics_epoch);
      pending_.pop_front();
    }
    return;
  }
  CHECK(!pending_.empty() && observations.size() == kv_lengths_.size());
  const auto& pending = pending_.front();
  DSparkSteadyStep step;
  step.generation = pending.generation;
  step.completed_ns = completed_ns;
  step.logical_requests = point_->batch.requests;
  step.draft_requests = point_->batch.requests;
  step.target_graph_rows = point_->batch.target_graph_rows;
  step.max_kv_length =
      *std::max_element(kv_lengths_.begin(), kv_lengths_.end()) +
      point_->draft_steps;
  step.request_ids = pending.request_ids;
  for (uint32_t row = 0; row < observations.size(); ++row) {
    const auto& observed = observations[row];
    CHECK_EQ(observed.draft_steps, point_->draft_steps);
    CHECK_LE(observed.accepted_drafts, observed.draft_steps);
    step.draft_tokens += observed.draft_steps;
    step.accepted_tokens += observed.accepted_drafts;
    kv_lengths_[row] += observed.accepted_drafts + 1;
  }
  CHECK_LE(step.max_kv_length, point_->batch.context.max_kv_length);
  if (const auto window = collector_->observe(step)) {
    windows_.emplace_back(*window);
  }
  pending_.pop_front();
}

std::vector<DSparkSteadyWindow> DraftStepStartupCalibration::finish_sample() {
  CHECK(point_ && pending_.empty() && collector_->finished());
  CHECK_EQ(windows_.size(), configuration_->windows_per_point);
  point_.reset();
  collector_.reset();
  return std::exchange(windows_, {});
}

void DraftStepStartupCalibration::install_costs(
    std::vector<std::vector<DSparkCostTableEntry>> costs) {
  CHECK(!sampling() && pending_.empty() && !selector_);
  CHECK_EQ(costs.size(), policy_.max_steps);
  policy_.costs = std::move(costs);
  selector_ = std::make_unique<DraftStepSelector>(std::move(policy_));
}

uint32_t DraftStepStartupCalibration::select(uint32_t requests,
                                             uint32_t max_kv_length) {
  if (!selector_) {
    return 0;
  }
  const auto draft = std::lower_bound(
      captured_requests_.begin(), captured_requests_.end(), requests);
  if (draft == captured_requests_.end()) {
    return 0;
  }
  std::vector<uint32_t> target_rows;
  target_rows.reserve(max_steps_);
  for (uint32_t depth = 1; depth <= max_steps_; ++depth) {
    const auto target =
        std::lower_bound(captured_graphs_.begin(),
                         captured_graphs_.end(),
                         static_cast<uint64_t>(requests) * (depth + 1));
    if (target == captured_graphs_.end()) {
      return 0;
    }
    target_rows.emplace_back(*target);
  }
  return selector_->select(requests, max_kv_length, *draft, target_rows);
}

}  // namespace xllm
