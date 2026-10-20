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

#include "core/framework/speculative/dspark/dspark_startup_calibration.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <nlohmann/json.hpp>

namespace xllm {
namespace {

nlohmann::json configuration() {
  return {{"model", "sps"},
          {"request_buckets", {1, 4}},
          {"contexts",
           {{{"prefill_tokens", 1868},
             {"min_kv_length", 1800},
             {"max_kv_length", 4096}}}}};
}

DSparkSteadyWindow window(uint32_t requests, uint32_t rows, double cost) {
  DSparkSteadyWindow result;
  result.first_generation = 1;
  result.last_generation = 33;
  result.logical_requests = requests;
  result.draft_requests = requests;
  result.target_graph_rows = rows;
  result.min_kv_length = 2000;
  result.max_kv_length = 2100;
  result.steps = 32;
  result.draft_tokens =
      static_cast<uint64_t>(std::min(requests * 7, rows - requests)) * 32;
  result.elapsed_ms = cost * 32;
  return result;
}

nlohmann::json sps_table(uint32_t maximum = 4096) {
  return {{"sample_batch_tokens", {1, 2, 4, 8, 16, 32}},
          {"sample_steps_per_sec", {100, 90, 80, 70, 60, 50}},
          {"max_batch_tokens", 32},
          {"max_kv_length", maximum}};
}

nlohmann::json bundle() {
  return {{"schema_version", 1},
          {"measurement", "steady_full_step"},
          {"compatibility",
           {{"model", "unit-test"}, {"quantization", "static-w8a8"}}},
          {"profiles",
           {{{"min_kv_length", 1},
             {"max_kv_length", 4096},
             {"request_buckets", {1, 4}},
             {"table", sps_table()}}}}};
}

TEST(DSparkStartupCalibrationTest, DefaultsUseAdditiveCostsAndBoundedContext) {
  const auto config = default_dspark_startup_calibration(202752);
  ASSERT_TRUE(config);
  EXPECT_EQ(config->model, DSparkSpsCostModel::Kind::ADDITIVE);
  EXPECT_TRUE(config->request_buckets.empty());
  EXPECT_TRUE(config->target_graph_rows.empty());
  ASSERT_EQ(config->contexts.size(), 1U);
  EXPECT_EQ(config->contexts.front().prefill_tokens, 2048U);
  EXPECT_EQ(config->contexts.front().min_kv_length, 1U);
  EXPECT_EQ(config->contexts.front().max_kv_length, 4096U);
  EXPECT_EQ(config->warmup_steps, 8U);
  EXPECT_EQ(config->window_steps, 32U);
  EXPECT_EQ(config->windows_per_point, 3U);
  EXPECT_EQ(config->token_seed, 1U);
}

TEST(DSparkStartupCalibrationTest, DefaultsRespectShortModelContexts) {
  auto config = default_dspark_startup_calibration(1024);
  ASSERT_TRUE(config);
  EXPECT_EQ(config->contexts.front().max_kv_length, 1024U);
  EXPECT_LT(config->contexts.front().prefill_tokens, 2048U);
  config->request_buckets = {1};
  EXPECT_TRUE(parse_dspark_startup_calibration(
      serialize_dspark_startup_calibration(*config)));
  EXPECT_FALSE(default_dspark_startup_calibration(864));
}

TEST(DSparkStartupCalibrationTest,
     EffectiveJsonIncludesOmittedDefaultsAndResolvedBuckets) {
  auto config = parse_dspark_startup_calibration(configuration().dump());
  ASSERT_TRUE(config);
  const auto data =
      nlohmann::json::parse(serialize_dspark_startup_calibration(*config));
  EXPECT_EQ(data.at("warmup_steps"), 8);
  EXPECT_EQ(data.at("window_steps"), 32);
  EXPECT_EQ(data.at("windows_per_point"), 3);
  EXPECT_EQ(data.at("token_seed"), 1);
  EXPECT_EQ(data.at("request_buckets"), nlohmann::json({1, 4}));
  EXPECT_EQ(data.at("contexts")[0].at("prefill_tokens"), 1868);
  EXPECT_TRUE(parse_dspark_startup_calibration(data.dump()));
  config->target_graph_rows = {1, 4, 8, 16, 32};
  EXPECT_EQ(nlohmann::json::parse(serialize_dspark_startup_calibration(*config))
                .at("target_graph_rows"),
            nlohmann::json({1, 4, 8, 16, 32}));
}

TEST(DSparkStartupCalibrationTest, ParsesConfigurationAndZeroSettling) {
  auto data = configuration();
  data["warmup_steps"] = 0;
  data["token_seed"] = 0;
  const auto config = parse_dspark_startup_calibration(data.dump());
  ASSERT_TRUE(config);
  EXPECT_EQ(config->model, DSparkSpsCostModel::Kind::SPS);
  EXPECT_EQ(config->warmup_steps, 0U);
  EXPECT_EQ(config->window_steps, 32U);
  EXPECT_EQ(config->windows_per_point, 3U);
  EXPECT_EQ(config->contexts.front().prefill_tokens, 1868U);
}

TEST(DSparkStartupCalibrationTest,
     RejectsInvalidConfigurationAndInsufficientKv) {
  for (const auto& patch :
       std::vector<nlohmann::json>{{{"model", "events"}},
                                   {{"request_buckets", {4, 1}}},
                                   {{"request_buckets", {1, 1}}},
                                   {{"request_buckets", {1, 129}}},
                                   {{"window_steps", 0}},
                                   {{"window_steps", true}},
                                   {{"warmup_steps", -1}},
                                   {{"windows_per_point", 4294967296ULL}},
                                   {{"contexts", nlohmann::json::array()}}}) {
    auto data = configuration();
    data.update(patch);
    EXPECT_FALSE(parse_dspark_startup_calibration(data.dump())) << data;
  }
  auto data = configuration();
  data["contexts"][0]["max_kv_length"] = 1900;
  EXPECT_FALSE(parse_dspark_startup_calibration(data.dump()));
  data = configuration();
  data["contexts"].emplace_back(data["contexts"][0]);
  EXPECT_FALSE(parse_dspark_startup_calibration(data.dump()));
}

TEST(DSparkStartupCalibrationTest,
     PlansActualCapturedFullWidthRatherThanEightB) {
  auto config = *parse_dspark_startup_calibration(configuration().dump());
  config.request_buckets = {1};
  const std::array<uint32_t, 2> requests{1, 4};
  const std::array<uint32_t, 6> graphs{1, 2, 4, 10, 16, 32};
  const auto points = plan_dspark_calibration(config, requests, graphs);
  ASSERT_TRUE(points);
  ASSERT_EQ(points->size(), 4U);
  EXPECT_EQ(points->back().target_graph_rows, 10U);
  EXPECT_EQ(points->back().requests, 1U);
}

TEST(DSparkStartupCalibrationTest, PlansKvCoverageBeforeFirstCapture) {
  EXPECT_EQ(dspark_calibration_warmup_length(4096, 32768, 128, 2048), 2048U);
  EXPECT_EQ(dspark_calibration_warmup_length(4097, 32768, 128, 2048), 2048U);
  EXPECT_EQ(dspark_calibration_warmup_length(4098, 32768, 128, 2048), 4091U);
  EXPECT_EQ(dspark_calibration_warmup_length(8192, 32768, 128, 2048), 4091U);
  EXPECT_EQ(dspark_calibration_warmup_length(32768, 32768, 128, 2048), 16379U);
  EXPECT_EQ(dspark_calibration_warmup_length(8192, 32768, 256, 16), 16U);
  EXPECT_EQ(dspark_calibration_warmup_length(8, 32, 1, 32), 16U);
  EXPECT_FALSE(dspark_calibration_warmup_length(0, 32768, 128, 2048));
  EXPECT_FALSE(dspark_calibration_warmup_length(32769, 32768, 128, 2048));
  EXPECT_FALSE(dspark_calibration_warmup_length(8200, 8200, 128, 2048));
  EXPECT_FALSE(dspark_calibration_warmup_length(8, 16, 1, 16));
  EXPECT_FALSE(dspark_calibration_warmup_length(4096, 32768, 0, 2048));
  EXPECT_FALSE(dspark_calibration_warmup_length(4096, 32768, 128, 0));
}

TEST(DSparkStartupCalibrationTest, RejectsUncapturedAndMissingFullWidthPoints) {
  auto config = *parse_dspark_startup_calibration(configuration().dump());
  const std::array<uint32_t, 2> requests{1, 4};
  const std::array<uint32_t, 6> graphs{1, 2, 4, 8, 16, 32};
  config.target_graph_rows = {1, 3, 8, 32};
  EXPECT_FALSE(plan_dspark_calibration(config, requests, graphs));
  config.target_graph_rows = {1, 2, 4, 8};
  EXPECT_FALSE(plan_dspark_calibration(config, requests, graphs));
  config.target_graph_rows.clear();
  config.request_buckets = {2};
  EXPECT_FALSE(plan_dspark_calibration(config, requests, graphs));
}

TEST(DSparkStartupCalibrationTest, SpsCellMediansMatchReferenceValues) {
  const std::vector<DSparkSteadyWindow> windows{window(1, 4, 10),
                                                window(1, 4, 12),
                                                window(1, 4, 100),
                                                window(2, 4, 20),
                                                window(1, 8, 30)};
  const auto fitted =
      fit_dspark_cost_table(windows, DSparkSpsCostModel::Kind::SPS, 1800, 4096);
  ASSERT_TRUE(fitted);
  const auto table = nlohmann::json::parse(*fitted);
  EXPECT_EQ(table["sample_batch_tokens"], nlohmann::json({4, 8}));
  EXPECT_DOUBLE_EQ(table["sample_steps_per_sec"][0].get<double>(), 62.5);
  EXPECT_DOUBLE_EQ(table["sample_steps_per_sec"][1].get<double>(),
                   1000.0 / 30.0);
  EXPECT_EQ(table["metadata"]["sampled_cells"], 3);
  EXPECT_EQ(table["metadata"]["cells"][0]["selected_draft_tokens"], 288);
  EXPECT_EQ(table["metadata"]["cells"][0]["generated_draft_tokens"], 672);
  EXPECT_EQ(table["metadata"]["cells"][0]["request_rounds"], 96);
  EXPECT_EQ(table["metadata"]["cells"][0]["sample_min_kv_length"], 2000);
  EXPECT_EQ(table["metadata"]["cells"][0]["sample_max_kv_length"], 2100);
}

TEST(DSparkStartupCalibrationTest, SpsProjectionWeightsCellsNotWindowCounts) {
  const std::vector<DSparkSteadyWindow> windows{
      window(1, 4, 10), window(1, 4, 12), window(1, 4, 100), window(2, 4, 200)};
  const auto fitted =
      fit_dspark_cost_table(windows, DSparkSpsCostModel::Kind::SPS, 1800, 4096);
  ASSERT_TRUE(fitted);
  const auto table = nlohmann::json::parse(*fitted);
  // Match the offline fitter: median(12, 200), not median(10, 12, 100, 200).
  EXPECT_DOUBLE_EQ(table["sample_steps_per_sec"][0].get<double>(),
                   1000.0 / 106.0);
}

TEST(DSparkStartupCalibrationTest, RejectsDisconnectedGridBeforeExecution) {
  auto config = *parse_dspark_startup_calibration(configuration().dump());
  config.model = DSparkSpsCostModel::Kind::ADDITIVE;
  const std::array<uint32_t, 3> requests{1, 4, 16};
  const std::array<uint32_t, 8> graphs{1, 2, 4, 8, 16, 32, 64, 128};
  config.request_buckets = {1, 16};
  EXPECT_FALSE(plan_dspark_calibration(config, requests, graphs));
  config.request_buckets = {1, 4, 16};
  EXPECT_TRUE(plan_dspark_calibration(config, requests, graphs));
}

TEST(DSparkStartupCalibrationTest, AdditiveRecoversConnectedAxesInSeconds) {
  const std::vector<DSparkSteadyWindow> windows{window(1, 1, 10),
                                                window(1, 2, 11),
                                                window(1, 4, 13),
                                                window(1, 8, 18),
                                                window(2, 2, 14),
                                                window(2, 4, 16),
                                                window(2, 8, 21),
                                                window(2, 16, 25),
                                                window(8, 8, 24),
                                                window(8, 16, 28),
                                                window(8, 32, 36),
                                                window(8, 64, 41)};
  const auto fitted = fit_dspark_cost_table(
      windows, DSparkSpsCostModel::Kind::ADDITIVE, 1800, 4096);
  ASSERT_TRUE(fitted);
  const auto table = nlohmann::json::parse(*fitted);
  EXPECT_NEAR(table["bias_seconds"].get<double>(), 0.010, 1e-12);
  EXPECT_NEAR(table["alpha_seconds"][1].get<double>(), 0.003, 1e-12);
  EXPECT_NEAR(table["alpha_seconds"][2].get<double>(), 0.006, 1e-12);
  EXPECT_LT(
      table["metadata"]["training_error"]["mean_absolute_ms"].get<double>(),
      1e-10);
}

TEST(DSparkStartupCalibrationTest,
     AdditiveRejectsDisconnectedAndNonpositiveGrid) {
  for (const auto& windows : std::vector<std::vector<DSparkSteadyWindow>>{
           {window(1, 1, 10), window(2, 2, 20)},
           {window(1, 1, 10), window(1, 4, 30), window(4, 4, 1)}}) {
    EXPECT_FALSE(fit_dspark_cost_table(
        windows, DSparkSpsCostModel::Kind::ADDITIVE, 1800, 4096));
  }
}

TEST(DSparkStartupCalibrationTest, FitRejectsIncompleteAndOutOfBucketWindows) {
  for (uint32_t fault = 0; fault < 5; ++fault) {
    auto sample = window(1, 8, 10);
    if (fault == 0) {
      sample.steps = 0;
    } else if (fault == 1) {
      sample.last_generation = 34;
    } else if (fault == 2) {
      sample.logical_requests = 2;
    } else if (fault == 3) {
      sample.max_kv_length = 4097;
    } else {
      sample.elapsed_ms = std::numeric_limits<double>::infinity();
    }
    EXPECT_FALSE(fit_dspark_cost_table(
        std::span(&sample, 1), DSparkSpsCostModel::Kind::SPS, 1800, 4096));
  }
}

TEST(DSparkStartupCalibrationTest,
     BundleSelectsContextWithoutCrossingCoverageGaps) {
  auto data = bundle();
  data["profiles"].emplace_back(nlohmann::json{{"min_kv_length", 5000},
                                               {"max_kv_length", 8192},
                                               {"request_buckets", {1}},
                                               {"table", sps_table(8192)}});
  const auto model = DSparkCostTableBundle::from_json(
      data.dump(), data["compatibility"].dump());
  ASSERT_TRUE(model);
  ASSERT_NE(model->lookup(4, 4096), nullptr);
  EXPECT_EQ(model->lookup(4, 4096)->max_kv_length, 4096U);
  EXPECT_EQ(model->lookup(1, 4500), nullptr);
  EXPECT_EQ(model->lookup(4, 6000), nullptr);
  ASSERT_NE(model->lookup(1, 6000), nullptr);
  EXPECT_EQ(model->lookup(1, 9000), nullptr);
  EXPECT_EQ(model->lookup(0, 1000), nullptr);
}

TEST(DSparkStartupCalibrationTest, BundleRejectsIncompatibleOrCorruptTables) {
  auto data = bundle();
  EXPECT_FALSE(DSparkCostTableBundle::from_json(data.dump(),
                                                R"({"model":"different"})"));
  for (const auto& patch :
       std::vector<nlohmann::json>{{{"schema_version", 2}},
                                   {{"schema_version", true}},
                                   {{"measurement", "device_events"}},
                                   {{"compatibility", nullptr}},
                                   {{"profiles", nlohmann::json::array()}}}) {
    auto invalid = data;
    invalid.update(patch);
    EXPECT_FALSE(DSparkCostTableBundle::from_json(invalid.dump())) << invalid;
  }
  data["profiles"].emplace_back(data["profiles"][0]);
  EXPECT_FALSE(DSparkCostTableBundle::from_json(data.dump()));
  data = bundle();
  data["profiles"][0]["table"]["max_kv_length"] = 8192;
  EXPECT_FALSE(DSparkCostTableBundle::from_json(data.dump()));
  data = bundle();
  data["profiles"][0]["request_buckets"] = {1, 129};
  EXPECT_FALSE(DSparkCostTableBundle::from_json(data.dump()));
}

TEST(DSparkStartupCalibrationTest,
     SessionCollectsIntervalsRatherThanEndpoints) {
  auto config = *parse_dspark_startup_calibration(configuration().dump());
  config.warmup_steps = 0;
  config.window_steps = 2;
  config.windows_per_point = 2;
  const DSparkCalibrationPoint point{1, 8, config.contexts.front()};
  DSparkCalibrationSession session(config, point);
  const std::vector<std::string> ids{"calibration-request"};
  DSparkSteadyStep step;
  step.logical_requests = 1;
  step.draft_requests = 1;
  step.target_graph_rows = 8;
  step.max_kv_length = 2000;
  step.draft_tokens = 7;
  step.accepted_tokens = 2;
  step.request_ids = ids;
  for (uint64_t generation = 1; generation <= 5; ++generation) {
    step.generation = generation;
    step.completed_ns = generation * 1000000;
    session.observe(step);
    EXPECT_EQ(session.finished(), generation == 5);
  }
  ASSERT_EQ(session.windows().size(), 2U);
  EXPECT_EQ(session.windows()[0].steps, 2U);
  EXPECT_EQ(session.windows()[0].accepted_tokens, 4U);
  EXPECT_DOUBLE_EQ(session.windows()[0].elapsed_ms, 2.0);
  EXPECT_EQ(session.windows()[1].first_generation, 3U);
  EXPECT_EQ(session.windows()[1].last_generation, 5U);
}

TEST(DSparkStartupCalibrationTest, SessionDiscardsTransientMembershipWindow) {
  auto config = *parse_dspark_startup_calibration(configuration().dump());
  config.warmup_steps = 0;
  config.window_steps = 2;
  config.windows_per_point = 1;
  DSparkCalibrationSession session(config, {1, 8, config.contexts.front()});
  const std::vector<std::string> ids{"calibration-request"};
  DSparkSteadyStep step;
  step.logical_requests = 1;
  step.draft_requests = 1;
  step.target_graph_rows = 8;
  step.max_kv_length = 2000;
  step.draft_tokens = 7;
  step.request_ids = ids;
  for (uint64_t generation = 1; generation <= 6; ++generation) {
    step.generation = generation;
    step.completed_ns = generation * 1000000;
    step.logical_requests = generation == 3 ? 2 : 1;
    session.observe(step);
  }
  ASSERT_TRUE(session.finished());
  ASSERT_EQ(session.windows().size(), 1U);
  EXPECT_EQ(session.windows()[0].first_generation, 4U);
  EXPECT_EQ(session.windows()[0].last_generation, 6U);
}

TEST(DSparkStartupCalibrationTest, LegacySingleTableRemainsLoadable) {
  const auto model = DSparkCostTableBundle::from_json(sps_table().dump());
  ASSERT_TRUE(model);
  ASSERT_NE(model->lookup(16, 2000), nullptr);
  EXPECT_EQ(model->lookup(1, 4097), nullptr);
  EXPECT_NEAR(
      *model->lookup(16, 2000)->model.step_time_ms(16, 32), 20.0, 1e-12);
}

}  // namespace
}  // namespace xllm
