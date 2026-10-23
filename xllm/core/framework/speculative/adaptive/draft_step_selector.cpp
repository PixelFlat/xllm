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

#include "core/framework/speculative/adaptive/draft_step_selector.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <utility>

namespace xllm {
namespace {

uint32_t read_count(const nlohmann::json& json,
                    const char* key,
                    uint32_t default_value) {
  const auto value = json.find(key);
  if (value == json.end()) {
    return default_value;
  }
  CHECK(value->is_number_integer()) << key << " must be an integer.";
  if (!value->is_number_unsigned()) {
    CHECK_GE(value->get<int64_t>(), 0) << key;
  }
  const uint64_t count = value->get<uint64_t>();
  CHECK_LE(count, std::numeric_limits<uint32_t>::max()) << key;
  return static_cast<uint32_t>(count);
}

double extrapolate_prefix_rate(std::span<const double> known) {
  CHECK(!known.empty());
  if (known.size() == 1) {
    return known.front();
  }
  if (known.front() <= 0.0) {
    return 0.0;
  }
  // Fit a geometric decay through the first and last known prefix rates.
  const double exponent = 1.0 / static_cast<double>(known.size() - 1);
  const double decay = std::pow(known.back() / known.front(), exponent);
  return std::clamp(known.back() * decay, 0.0, known.back());
}

}  // namespace

DraftStepSelectorConfig load_draft_step_selector_config(const std::string& path,
                                                        uint32_t max_steps) {
  DraftStepSelectorConfig config;
  config.max_steps = max_steps;
  if (path.empty()) {
    return config;
  }
  std::ifstream stream(path);
  CHECK(stream.is_open()) << "Cannot open adaptive draft-step configuration: "
                          << path;
  const nlohmann::json json = nlohmann::json::parse(stream);
  CHECK(json.is_object());
  CHECK_EQ(read_count(json, "version", /*default_value=*/0), 1U);
  CHECK(!json.contains("step_cost_models"))
      << "Draft-step costs are measured at startup, not supplied as JSON.";
  const auto& policy = json.contains("policy") ? json.at("policy") : json;
  CHECK(policy.is_object());
  CHECK(!policy.contains("min_observations") &&
        !policy.contains("exploration_interval"))
      << "Draft-step statistics use full completed-batch windows and geometric "
         "extrapolation; configure window_size and update_interval instead.";
  config.window_size = read_count(policy, "window_size", config.window_size);
  config.update_interval =
      read_count(policy, "update_interval", config.update_interval);
  config.switch_min_gain =
      policy.value("switch_min_gain", config.switch_min_gain);
  LOG(INFO) << "Adaptive draft-step configuration: " << json.dump()
            << ", max_steps=" << max_steps << ", window=" << config.window_size
            << ", update_interval=" << config.update_interval
            << ", switch_min_gain=" << config.switch_min_gain;
  return config;
}

DraftStepSelector::DraftStepSelector(DraftStepSelectorConfig config)
    : config_(std::move(config)), current_steps_(config_.max_steps) {
  CHECK_GT(config_.max_steps, 0U);
  CHECK_LE(config_.max_steps, 64U);
  CHECK_GT(config_.window_size, 0U);
  CHECK_LE(config_.window_size, 1048576U);
  CHECK_GT(config_.update_interval, 0U);
  CHECK(std::isfinite(config_.switch_min_gain));
  CHECK_GE(config_.switch_min_gain, 0.0);
  CHECK_EQ(config_.costs.size(), config_.max_steps);
  for (const auto& profiles : config_.costs) {
    CHECK(!profiles.empty());
    for (const auto& profile : profiles) {
      CHECK_GT(profile.min_kv_length, 0U);
      CHECK_GE(profile.max_kv_length, profile.min_kv_length);
    }
  }
  histories_.resize(config_.max_steps);
  acceptance_sums_.resize(config_.max_steps, 0.0);
  minimum_epochs_.resize(config_.max_steps, 0);
}

void DraftStepSelector::observe(
    std::span<const DraftStepAcceptanceObservation> observations,
    uint64_t statistics_epoch) {
  if (observations.empty()) {
    return;
  }
  CHECK_LE(statistics_epoch, statistics_epoch_);
  const uint32_t depth = observations.front().draft_steps;
  CHECK_GT(depth, 0U);
  CHECK_LE(depth, config_.max_steps);
  const auto eligible_end = std::upper_bound(
      minimum_epochs_.begin(), minimum_epochs_.end(), statistics_epoch);
  const uint32_t tracked = std::min(
      depth, static_cast<uint32_t>(eligible_end - minimum_epochs_.begin()));
  CHECK_GT(tracked, 0U);
  std::vector<uint64_t> counts(tracked + 1, /*value=*/0);
  for (const auto& observation : observations) {
    CHECK_EQ(observation.draft_steps, depth)
        << "One draft-step observation span must contain one verify batch.";
    CHECK_LE(observation.accepted_drafts, observation.draft_steps);
    ++counts[std::min(observation.accepted_drafts, tracked)];
  }
  uint64_t remaining = static_cast<uint64_t>(observations.size());
  for (uint32_t position = 0; position < tracked; ++position) {
    remaining -= counts[position];
    const double rate = static_cast<double>(remaining) /
                        static_cast<double>(observations.size());
    auto& history = histories_[position];
    if (history.size() == config_.window_size) {
      acceptance_sums_[position] -= history.front();
      history.pop_front();
    }
    history.emplace_back(rate);
    acceptance_sums_[position] += rate;
  }
  ++completed_batches_;
}

std::vector<double> DraftStepSelector::position_acceptance() const {
  std::vector<double> rates;
  rates.reserve(config_.max_steps);
  if (histories_.front().empty()) {
    return rates;
  }
  for (uint32_t position = 0; position < config_.max_steps; ++position) {
    const auto& history = histories_[position];
    const double rate =
        history.empty() ? extrapolate_prefix_rate(rates)
                        : std::clamp(acceptance_sums_[position] /
                                         static_cast<double>(history.size()),
                                     0.0,
                                     1.0);
    // Each entry is P(the whole prefix through j is accepted), not P(token j
    // is accepted | the earlier tokens were accepted). Do not multiply again.
    rates.emplace_back(rate);
  }
  return rates;
}

void DraftStepSelector::clear_positions_above(uint32_t steps) {
  CHECK_LT(statistics_epoch_, std::numeric_limits<uint64_t>::max());
  ++statistics_epoch_;
  for (uint32_t position = steps; position < config_.max_steps; ++position) {
    histories_[position].clear();
    acceptance_sums_[position] = 0.0;
    minimum_epochs_[position] = statistics_epoch_;
  }
}

uint32_t DraftStepSelector::select(
    uint32_t batch_size,
    uint32_t max_kv_length,
    uint32_t draft_requests,
    std::span<const uint32_t> target_graph_rows) {
  CHECK_GT(batch_size, 0U);
  CHECK_GT(max_kv_length, 0U);
  CHECK_GE(draft_requests, batch_size);
  CHECK_EQ(target_graph_rows.size(), config_.max_steps);
  std::vector<double> costs;
  costs.reserve(config_.max_steps);
  for (uint32_t index = 0; index < config_.max_steps; ++index) {
    const auto& profiles = config_.costs[index];
    const auto profile =
        std::find_if(profiles.begin(), profiles.end(), [&](const auto& entry) {
          return entry.min_kv_length <= max_kv_length &&
                 entry.max_kv_length >= max_kv_length &&
                 (entry.request_buckets.empty() ||
                  std::find(entry.request_buckets.begin(),
                            entry.request_buckets.end(),
                            draft_requests) != entry.request_buckets.end());
        });
    const auto cost = profile == profiles.end()
                          ? std::optional<double>()
                          : profile->model.step_time_ms(
                                draft_requests, target_graph_rows[index]);
    if (!cost) {
      // Missing profile coverage is decided before any task is submitted.
      return config_.max_steps;
    }
    costs.emplace_back(*cost);
  }
  if (completed_batches_ - last_update_ < config_.update_interval) {
    return current_steps_;
  }
  last_update_ = completed_batches_;
  const auto active_end = histories_.begin() + current_steps_;
  if (!std::all_of(histories_.begin(), active_end, [&](const auto& history) {
        return history.size() == config_.window_size;
      })) {
    return current_steps_;
  }
  const auto rates = position_acceptance();
  CHECK_EQ(rates.size(), config_.max_steps);
  std::vector<double> scores;
  scores.reserve(config_.max_steps);
  double expected = 1.0;
  for (uint32_t position = 0; position < config_.max_steps; ++position) {
    expected += rates[position];
    scores.emplace_back(batch_size * expected / costs[position]);
  }
  const uint32_t best =
      static_cast<uint32_t>(std::max_element(scores.begin(), scores.end()) -
                            scores.begin()) +
      1;
  const double required_score =
      scores[current_steps_ - 1] * (1.0 + config_.switch_min_gain);
  const bool switch_steps =
      best != current_steps_ && scores[best - 1] > required_score;
  if (switch_steps) {
    if (best < current_steps_) {
      clear_positions_above(best);
    }
    current_steps_ = best;
  }
  return current_steps_;
}

}  // namespace xllm
