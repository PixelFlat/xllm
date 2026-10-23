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
#include <gtest/gtest.h>

#include <array>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/framework/speculative/draft_step_startup_calibration.h"

namespace xllm {
namespace {

class DraftStepDecisionTraceSink final : public google::LogSink {
 public:
  DraftStepDecisionTraceSink()
      : previous_level_(google::SetVLOGLevel("draft_step_selector", 1)) {
    google::AddLogSink(this);
  }

  ~DraftStepDecisionTraceSink() override {
    google::RemoveLogSink(this);
    google::SetVLOGLevel("draft_step_selector", previous_level_);
  }

  void send(google::LogSeverity /*severity*/,
            const char* /*full_filename*/,
            const char* /*base_filename*/,
            int /*line*/,
            const google::LogMessageTime& /*time*/,
            const char* message,
            size_t message_length) override {
    constexpr std::string_view prefix = "Adaptive draft-step decision: ";
    const std::string_view text(message, message_length);
    if (!text.starts_with(prefix)) {
      return;
    }
    std::lock_guard lock(mutex_);
    records_.emplace_back(text.substr(prefix.size()));
  }

  std::vector<nlohmann::json> snapshots() {
    std::lock_guard lock(mutex_);
    std::vector<nlohmann::json> result;
    result.reserve(records_.size());
    for (const auto& record : records_) {
      result.emplace_back(nlohmann::json::parse(record));
    }
    return result;
  }

 private:
  int32_t previous_level_;
  std::mutex mutex_;
  std::vector<std::string> records_;
};

DraftStepSelectorConfig policy(std::vector<double> milliseconds) {
  DraftStepSelectorConfig config;
  config.max_steps = static_cast<uint32_t>(milliseconds.size());
  config.window_size = 1;
  config.update_interval = 1;
  config.switch_min_gain = 0.0;
  config.costs.reserve(milliseconds.size());
  for (const double ms : milliseconds) {
    const std::string text =
        "{\"sample_batch_tokens\":[1],\"sample_steps_per_sec\":[" +
        std::to_string(1000.0 / ms) +
        "],\"max_batch_tokens\":1024,\"max_kv_length\":4096}";
    auto cost = DSparkSpsCostModel::from_json(text);
    CHECK(cost.has_value());
    config.costs.emplace_back(
        std::vector<DSparkCostTableEntry>{{1, 4096, {}, std::move(*cost)}});
  }
  return config;
}

constexpr std::array<uint32_t, 3> kRows{2, 4, 4};
constexpr std::array<uint32_t, 5> kFiveStepRows{2, 4, 4, 8, 8};

TEST(DraftStepSelectorTest, ColdStartPreservesConfiguredMaximum) {
  DraftStepSelector selector(policy({1.0, 2.0, 4.0}));
  EXPECT_TRUE(selector.position_acceptance().empty());
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 3U);
}

TEST(DraftStepSelectorTest, MaximizesExpectedTokensOverWholeCycleCost) {
  DraftStepSelector selector(policy({1.6, 1.7, 3.0}));
  const std::array<DraftStepAcceptanceObservation, 2> history{{{3, 2}, {3, 1}}};
  selector.observe(history, selector.statistics_epoch());
  EXPECT_EQ(selector.position_acceptance(),
            std::vector<double>({1.0, 0.5, 0.0}));
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 2U);
}

TEST(DraftStepSelectorTest, DoesNotMultiplyPrefixProbabilitiesTwice) {
  DraftStepSelector selector(policy({1.0, 1.4, 1.5}));
  const std::array<DraftStepAcceptanceObservation, 2> history{{{3, 3}, {3, 0}}};
  selector.observe(history, selector.statistics_epoch());
  EXPECT_EQ(selector.position_acceptance(),
            std::vector<double>({0.5, 0.5, 0.5}));
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 3U);
}

TEST(DraftStepSelectorTest, FiveStepPolicyKeepsGainBelowDefaultMargin) {
  DraftStepDecisionTraceSink trace;
  auto config = policy({100.0, 200.0, 200.0, 200.0, 104.0});
  config.switch_min_gain = DraftStepSelectorConfig{}.switch_min_gain;
  DraftStepSelector selector(std::move(config));
  const std::array<DraftStepAcceptanceObservation, 1> history{{{5, 0}}};
  selector.observe(history, selector.statistics_epoch());
  EXPECT_EQ(selector.select(1, 128, 1, kFiveStepRows), 5U);
  const auto records = trace.snapshots();
  ASSERT_EQ(records.size(), 1U);
  const auto& decision = records.front();
  EXPECT_EQ(decision.at("reason"), "below_switch_margin");
  EXPECT_EQ(decision.at("raw_best_steps"), 1);
  EXPECT_EQ(decision.at("selected_steps"), 5);
  EXPECT_EQ(decision.at("prefix_acceptance"),
            nlohmann::json::array({0.0, 0.0, 0.0, 0.0, 0.0}));
  const auto& candidates = decision.at("candidates");
  ASSERT_EQ(candidates.size(), 5U);
  EXPECT_DOUBLE_EQ(candidates.front().at("cost_ms").get<double>(), 100.0);
  EXPECT_DOUBLE_EQ(
      candidates.front().at("expected_tokens_per_request").get<double>(), 1.0);
  EXPECT_DOUBLE_EQ(candidates.front().at("score_tok_ms").get<double>(), 0.01);
  EXPECT_EQ(candidates.back().at("target_graph_rows"), 8);
}

TEST(DraftStepSelectorTest, FiveStepPolicySwitchesAboveDefaultMargin) {
  DraftStepDecisionTraceSink trace;
  auto config = policy({100.0, 200.0, 200.0, 200.0, 106.0});
  config.switch_min_gain = DraftStepSelectorConfig{}.switch_min_gain;
  DraftStepSelector selector(std::move(config));
  const std::array<DraftStepAcceptanceObservation, 1> history{{{5, 0}}};
  selector.observe(history, selector.statistics_epoch());
  EXPECT_EQ(selector.select(1, 128, 1, kFiveStepRows), 1U);
  const auto records = trace.snapshots();
  ASSERT_EQ(records.size(), 1U);
  EXPECT_EQ(records.front().at("reason"), "switch");
  EXPECT_EQ(records.front().at("current_steps"), 5);
  EXPECT_EQ(records.front().at("selected_steps"), 1);
}

TEST(DraftStepSelectorTest, OneBatchContributesOnePrefixRatePerPosition) {
  DraftStepSelector selector(policy({1.0, 2.0, 4.0}));
  const std::array<DraftStepAcceptanceObservation, 4> history{
      {{3, 3}, {3, 2}, {3, 1}, {3, 0}}};
  selector.observe(history, selector.statistics_epoch());
  EXPECT_EQ(selector.position_acceptance(),
            std::vector<double>({0.75, 0.5, 0.25}));
}

TEST(DraftStepSelectorTest, DifferentBatchSizesHaveEqualWindowWeight) {
  auto config = policy({1.0, 2.0, 4.0});
  config.window_size = 2;
  DraftStepSelector selector(std::move(config));
  const std::array<DraftStepAcceptanceObservation, 1> small{{{3, 3}}};
  const std::array<DraftStepAcceptanceObservation, 3> large{
      {{3, 0}, {3, 0}, {3, 0}}};
  selector.observe(small, selector.statistics_epoch());
  selector.observe(large, selector.statistics_epoch());
  EXPECT_EQ(selector.position_acceptance(),
            std::vector<double>({0.5, 0.5, 0.5}));
}

TEST(DraftStepSelectorTest, WindowExpiresOldBatchRates) {
  auto config = policy({1.0, 2.0, 4.0});
  config.window_size = 2;
  DraftStepSelector selector(std::move(config));
  const std::array<DraftStepAcceptanceObservation, 1> accepted{{{3, 3}}};
  const std::array<DraftStepAcceptanceObservation, 1> rejected{{{3, 0}}};
  selector.observe(accepted, selector.statistics_epoch());
  selector.observe(rejected, selector.statistics_epoch());
  EXPECT_EQ(selector.position_acceptance(),
            std::vector<double>({0.5, 0.5, 0.5}));
  selector.observe(rejected, selector.statistics_epoch());
  EXPECT_EQ(selector.position_acceptance(),
            std::vector<double>({0.0, 0.0, 0.0}));
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 1U);
}

TEST(DraftStepSelectorTest, ColdStartRequiresFullCompletedBatchWindows) {
  auto config = policy({1.0, 2.0, 4.0});
  config.window_size = 20;
  DraftStepSelector selector(std::move(config));
  const std::vector<DraftStepAcceptanceObservation> large(128, {3, 0});
  for (uint32_t batch = 0; batch < 19; ++batch) {
    selector.observe(large, selector.statistics_epoch());
    EXPECT_EQ(selector.select(1, 128, 1, kRows), 3U);
  }
  selector.observe(large, selector.statistics_epoch());
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 1U);
}

TEST(DraftStepSelectorTest,
     ExtrapolatesSingleKnownPositionWithoutRejectingTail) {
  DraftStepSelector selector(policy({1.0, 2.0, 4.0}));
  const std::array<DraftStepAcceptanceObservation, 2> history{{{1, 1}, {1, 0}}};
  selector.observe(history, selector.statistics_epoch());
  EXPECT_EQ(selector.position_acceptance(),
            std::vector<double>({0.5, 0.5, 0.5}));
}

TEST(DraftStepSelectorTest, ExtrapolatesMissingPositionsGeometrically) {
  DraftStepSelector selector(policy({1.0, 2.0, 4.0}));
  const std::array<DraftStepAcceptanceObservation, 4> history{
      {{2, 2}, {2, 1}, {2, 1}, {2, 0}}};
  selector.observe(history, selector.statistics_epoch());
  const auto rates = selector.position_acceptance();
  ASSERT_EQ(rates.size(), 3U);
  EXPECT_DOUBLE_EQ(rates[0], 0.75);
  EXPECT_DOUBLE_EQ(rates[1], 0.25);
  EXPECT_NEAR(rates[2], 1.0 / 12.0, 1e-12);
}

TEST(DraftStepSelectorTest, ZeroAcceptanceHasFiniteZeroExtrapolation) {
  DraftStepSelector selector(policy({1.0, 2.0, 4.0}));
  const std::array<DraftStepAcceptanceObservation, 1> history{{{2, 0}}};
  selector.observe(history, selector.statistics_epoch());
  EXPECT_EQ(selector.position_acceptance(),
            std::vector<double>({0.0, 0.0, 0.0}));
}

TEST(DraftStepSelectorTest, DecreasingDepthClearsDeeperWindows) {
  DraftStepSelector selector(policy({1.6, 1.7, 100.0}));
  const std::array<DraftStepAcceptanceObservation, 2> history{{{3, 3}, {3, 1}}};
  selector.observe(history, selector.statistics_epoch());
  EXPECT_EQ(selector.position_acceptance(),
            std::vector<double>({1.0, 0.5, 0.5}));
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 2U);
  EXPECT_EQ(selector.position_acceptance(),
            std::vector<double>({1.0, 0.5, 0.25}));
}

TEST(DraftStepSelectorTest, InFlightOlderBatchCannotRefillClearedPositions) {
  DraftStepSelector selector(policy({1.6, 1.7, 100.0}));
  const uint64_t old_epoch = selector.statistics_epoch();
  const std::array<DraftStepAcceptanceObservation, 2> initial{{{3, 3}, {3, 1}}};
  selector.observe(initial, old_epoch);
  ASSERT_EQ(selector.select(1, 128, 1, kRows), 2U);
  ASSERT_GT(selector.statistics_epoch(), old_epoch);
  const std::array<DraftStepAcceptanceObservation, 2> delayed{{{3, 2}, {3, 1}}};
  selector.observe(delayed, old_epoch);
  EXPECT_EQ(selector.position_acceptance(),
            std::vector<double>({1.0, 0.5, 0.25}));
  // A later full-depth batch (e.g. explicit out-of-coverage admission) may
  // contribute fresh deep measurements again without periodic probing.
  ASSERT_EQ(selector.select(1, 4097, 1, kRows), 3U);
  selector.observe(initial, selector.statistics_epoch());
  EXPECT_EQ(selector.position_acceptance(),
            std::vector<double>({1.0, 0.5, 0.5}));
}

TEST(DraftStepSelectorTest, ReexpansionWaitsForNewDeepWindowsBeforeRescoring) {
  DraftStepSelector selector(policy({1.6, 1.7, 2.0}));
  const std::array<DraftStepAcceptanceObservation, 2> initial{{{3, 2}, {3, 1}}};
  selector.observe(initial, selector.statistics_epoch());
  ASSERT_EQ(selector.select(1, 128, 1, kRows), 2U);
  const std::array<DraftStepAcceptanceObservation, 1> accepted{{{2, 2}}};
  selector.observe(accepted, selector.statistics_epoch());
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 3U);
  const std::array<DraftStepAcceptanceObservation, 1> earlier_shallow{{{2, 0}}};
  selector.observe(earlier_shallow, selector.statistics_epoch());
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 3U);
  const std::array<DraftStepAcceptanceObservation, 1> rejected{{{3, 0}}};
  selector.observe(rejected, selector.statistics_epoch());
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 1U);
}

TEST(DraftStepSelectorTest, ExplicitlyUsesMaximumOutsideCostCoverage) {
  DraftStepSelector selector(policy({1.0, 2.0, 4.0}));
  const std::array<DraftStepAcceptanceObservation, 1> history{{{3, 0}}};
  selector.observe(history, selector.statistics_epoch());
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 1U);
  EXPECT_EQ(selector.select(1, 4097, 1, kRows), 3U);
  const std::array<uint32_t, 3> uncovered{1025, 1026, 1027};
  EXPECT_EQ(selector.select(1, 128, 1, uncovered), 3U);
}

TEST(DraftStepSelectorTest,
     DecisionCadenceCountsCompletedBatchesNotSelections) {
  auto config = policy({1.0, 2.0, 4.0});
  config.update_interval = 2;
  DraftStepSelector selector(std::move(config));
  const std::array<DraftStepAcceptanceObservation, 1> history{{{3, 0}}};
  selector.observe({}, selector.statistics_epoch());
  selector.observe(history, selector.statistics_epoch());
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 3U);
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 3U);
  selector.observe(history, selector.statistics_epoch());
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 1U);
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 1U);
}

TEST(DraftStepSelectorTest, ReusesAdditivePhysicalBatchAndTokenAxes) {
  auto config = policy({1.0, 2.0, 4.0});
  config.costs.clear();
  for (
      const std::string& text :
      {R"({"bias_seconds":0.001,"bs_probes":[1,8],"alpha_seconds":[0,0.02],"m_probes":[1,64],"theta_seconds":[0,0]})",
       R"({"bias_seconds":0.003,"bs_probes":[1,8],"alpha_seconds":[0,0],"m_probes":[1,64],"theta_seconds":[0,0]})",
       R"({"bias_seconds":0.1,"bs_probes":[1,8],"alpha_seconds":[0,0],"m_probes":[1,64],"theta_seconds":[0,0]})"}) {
    auto cost = DSparkSpsCostModel::from_json(text);
    ASSERT_TRUE(cost.has_value());
    config.costs.emplace_back(
        std::vector<DSparkCostTableEntry>{{1, 4096, {}, std::move(*cost)}});
  }
  DraftStepSelector selector(std::move(config));
  const std::array<DraftStepAcceptanceObservation, 1> history{{{3, 0}}};
  selector.observe(history, selector.statistics_epoch());
  EXPECT_EQ(selector.select(1, 128, 1, kRows), 1U);
  const std::array<uint32_t, 3> padded{16, 32, 32};
  const std::array<DraftStepAcceptanceObservation, 1> shallow{{{1, 0}}};
  selector.observe(shallow, selector.statistics_epoch());
  EXPECT_EQ(selector.select(8, 128, 8, padded), 2U);
}

TEST(DraftStepSelectorTest, RejectsMixingDifferentDepthsIntoOneBatchSample) {
  DraftStepSelector selector(policy({1.0, 2.0, 4.0}));
  const std::array<DraftStepAcceptanceObservation, 2> mixed{{{3, 3}, {1, 1}}};
  EXPECT_DEATH_IF_SUPPORTED(
      selector.observe(mixed, selector.statistics_epoch()), "one verify batch");
}

TEST(DraftStepStartupCalibrationTest,
     DefaultsMeasureCyclesRatherThanLoadCostJson) {
  const auto config = load_draft_step_selector_config("", /*max_steps=*/3);
  EXPECT_TRUE(config.costs.empty());
  EXPECT_EQ(config.window_size, 20U);
  EXPECT_EQ(config.update_interval, 5U);
  EXPECT_DOUBLE_EQ(config.switch_min_gain, 0.05);
  DraftStepStartupCalibration calibration("", /*max_steps=*/3);
  const auto& sampling = calibration.configuration(/*max_positions=*/8192);
  EXPECT_EQ(sampling.draft_width, 3U);
  EXPECT_EQ(sampling.model, DSparkSpsCostModel::Kind::ADDITIVE);
  calibration.set_geometry({1, 2}, {2, 4, 8});
  const auto points = calibration.points();
  ASSERT_EQ(points.size(), 11U);
  for (const auto& point : points) {
    EXPECT_GE(point.batch.target_graph_rows,
              point.batch.requests * (point.draft_steps + 1));
  }
  EXPECT_EQ(calibration.select(1, 128), 0U);
}

TEST(DraftStepStartupCalibrationTest,
     MeasuresActualDepthThroughOrdinaryResultFences) {
  DraftStepStartupCalibration calibration("", /*max_steps=*/3);
  const auto& config = calibration.configuration(/*max_positions=*/8192);
  calibration.set_geometry({1, 2}, {2, 4, 8});
  const auto point = calibration.points().front();
  ASSERT_EQ(point.draft_steps, 1U);
  calibration.begin_sample(point);
  ASSERT_TRUE(calibration.sampling());
  EXPECT_EQ(calibration.forced_steps(), 1U);
  EXPECT_EQ(calibration.forced_graph_rows(), 2U);
  const std::array<std::string, 1> ids{"sample-0"};
  const std::array<DraftStepAcceptanceObservation, 1> observed{{{1, 1}}};
  const uint32_t rounds =
      config.warmup_steps + config.window_steps * config.windows_per_point + 1;
  for (uint32_t round = 0; round < rounds; ++round) {
    calibration.dispatched(ids);
    calibration.observe(observed, static_cast<uint64_t>(round + 1) * 1000000);
  }
  const auto windows = calibration.finish_sample();
  ASSERT_EQ(windows.size(), config.windows_per_point);
  EXPECT_FALSE(calibration.sampling());
  const auto fitted =
      fit_draft_step_cost_table(windows,
                                DSparkSpsCostModel::Kind::ADDITIVE,
                                point.batch.context.min_kv_length,
                                point.batch.context.max_kv_length,
                                /*draft_width=*/1);
  ASSERT_TRUE(fitted);
  const auto model = DSparkSpsCostModel::from_json(*fitted);
  ASSERT_TRUE(model);
  ASSERT_TRUE(model->step_time_ms(/*draft_requests=*/1, /*target_rows=*/2));
  EXPECT_DOUBLE_EQ(*model->step_time_ms(1, 2), 1.0);
  // Reusing the collector/fitter does not relax the DSpark Draft5/7 guard.
  EXPECT_FALSE(fit_dspark_cost_table(windows,
                                     DSparkSpsCostModel::Kind::ADDITIVE,
                                     point.batch.context.min_kv_length,
                                     point.batch.context.max_kv_length,
                                     /*draft_width=*/1));
}

TEST(DraftStepStartupCalibrationTest,
     ServingStartsOnlyAfterMeasuredModelsInstalled) {
  DraftStepStartupCalibration calibration("", /*max_steps=*/3);
  calibration.configuration(/*max_positions=*/8192);
  calibration.set_geometry({1, 2}, {2, 4, 8});
  auto config = policy({1.0, 2.0, 4.0});
  calibration.install_costs(std::move(config.costs));
  EXPECT_EQ(calibration.select(1, 128), 3U);
  const std::array<std::string, 1> ids{"request-0"};
  const std::array<DraftStepAcceptanceObservation, 1> observed{{{3, 0}}};
  calibration.dispatched(ids);
  calibration.observe(observed, /*completed_ns=*/1);
  EXPECT_EQ(calibration.select(1, 128), 3U);
  EXPECT_EQ(calibration.select(3, 128), 0U);
}

TEST(DraftStepStartupCalibrationTest,
     RetiredBatchesKeepTheirDispatchStatisticsEpoch) {
  DraftStepStartupCalibration calibration("", /*max_steps=*/3);
  calibration.configuration(/*max_positions=*/8192);
  calibration.set_geometry({1, 2}, {2, 4, 8});
  auto config = policy({1.6, 1.7, 2.0});
  calibration.install_costs(std::move(config.costs));
  const std::array<std::string, 2> ids{"request-0", "request-1"};
  const std::array<DraftStepAcceptanceObservation, 2> deep{{{3, 2}, {3, 1}}};
  const std::array<DraftStepAcceptanceObservation, 2> shallow{{{2, 2}, {2, 1}}};
  const std::array<DraftStepAcceptanceObservation, 2> delayed{{{3, 3}, {3, 3}}};
  uint64_t completed_ns = 0;
  const auto submit = [&](uint32_t expected) {
    EXPECT_EQ(calibration.select(2, 128), expected);
    calibration.dispatched(ids);
  };
  for (uint32_t batch = 0; batch < 18; ++batch) {
    submit(/*expected=*/3);
    calibration.observe(deep, ++completed_ns);
  }
  submit(/*expected=*/3);
  submit(/*expected=*/3);
  calibration.observe(deep, ++completed_ns);
  submit(/*expected=*/3);
  calibration.observe(deep, ++completed_ns);
  submit(/*expected=*/2);
  // This full-acceptance result belongs to a pre-shrink task. Its third
  // position must not become a fresh p[2]=1 sample after the window was reset.
  calibration.observe(delayed, ++completed_ns);
  calibration.observe(shallow, ++completed_ns);
  for (uint32_t batch = 0; batch < 3; ++batch) {
    submit(/*expected=*/2);
    calibration.observe(shallow, ++completed_ns);
  }
  EXPECT_EQ(calibration.select(2, 128), 2U);
}

}  // namespace
}  // namespace xllm
