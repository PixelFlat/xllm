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
#include <string_view>
#include <vector>

#include "core/framework/speculative/dspark/dspark_sps_cost_model.h"
#include "core/framework/speculative/dspark/dspark_window_collector.h"

namespace xllm {

// Startup control-plane messages, never issued between serving decode rounds.
enum class DSparkCalibrationAction : uint8_t {
  CAPABILITIES = 1,
  BEGIN_SAMPLE = 2,
  FINISH_SAMPLE = 3,
  INSTALL_COSTS = 4,
};

struct DSparkCalibrationCommand {
  DSparkCalibrationAction action = DSparkCalibrationAction::CAPABILITIES;
  std::string payload;
};

struct DSparkCalibrationContext {
  uint32_t prefill_tokens = 0;
  uint32_t min_kv_length = 1;
  uint32_t max_kv_length = 0;
};

struct DSparkStartupCalibrationConfig {
  DSparkSpsCostModel::Kind model = DSparkSpsCostModel::Kind::SPS;
  std::vector<uint32_t> request_buckets;
  std::vector<uint32_t> target_graph_rows;
  std::vector<DSparkCalibrationContext> contexts;
  uint32_t warmup_steps = 8;
  uint32_t window_steps = 32;
  uint32_t windows_per_point = 3;
  uint32_t token_seed = 1;
  uint32_t draft_width = 7;
};

struct DSparkCalibrationPoint {
  uint32_t requests = 0;
  uint32_t target_graph_rows = 0;
  DSparkCalibrationContext context;
};

// Whole-round graphs bind actual B as well as G. An empty coverage list keeps
// the split-graph policy; otherwise each listed request bucket is explicit.
struct DSparkCalibrationGraphCoverage {
  uint32_t requests = 0;
  std::vector<uint32_t> target_graph_rows;
};

std::optional<DSparkStartupCalibrationConfig> parse_dspark_startup_calibration(
    std::string_view contents,
    uint32_t draft_width = 7);

// Default contexts are bounded by the model; request buckets are resolved from
// the actual captured geometry after warmup, not guessed from HTTP concurrency.
std::optional<DSparkStartupCalibrationConfig>
default_dspark_startup_calibration(uint32_t max_positions,
                                   uint32_t draft_width = 7);

// Materialize numeric defaults; an omitted target axis selects legal captures.
std::string serialize_dspark_startup_calibration(
    const DSparkStartupCalibrationConfig& config);

// Keep the legacy bound when compact-page rounding already covers the sample;
// otherwise choose the smallest sufficient capture. Both in-flight allocations
// of draft_width + 1 tokens must fit the model limit.
std::optional<uint32_t> dspark_calibration_warmup_length(
    uint32_t max_kv_length,
    uint32_t max_positions,
    uint32_t logical_block_size,
    uint32_t default_length,
    uint32_t draft_width = 7);

// Only captured physical buckets are legal. Full width is the first captured
// bucket covering (draft_width + 1)*Bcap, which need not be that exact size.
std::optional<std::vector<DSparkCalibrationPoint>> plan_dspark_calibration(
    const DSparkStartupCalibrationConfig& config,
    std::span<const uint32_t> captured_requests,
    std::span<const uint32_t> captured_target_rows,
    std::span<const DSparkCalibrationGraphCoverage> graph_coverage = {});

// Host-only startup fit using cell medians and an identifiable additive model.
// No device/runtime or subprocess is involved. The serialized table includes
// training diagnostics.
std::optional<std::string> fit_dspark_cost_table(
    std::span<const DSparkSteadyWindow> windows,
    DSparkSpsCostModel::Kind kind,
    uint32_t min_kv_length,
    uint32_t max_kv_length,
    uint32_t draft_width = 7);

// The owner starts/replaces a sample only after draining the pipeline.
// observe() runs on the existing result-retirement thread, without file I/O.
class DSparkCalibrationSession final {
 public:
  DSparkCalibrationSession(const DSparkStartupCalibrationConfig& config,
                           DSparkCalibrationPoint point);
  void observe(const DSparkSteadyStep& step);
  bool finished() const;
  std::span<const DSparkSteadyWindow> windows() const { return windows_; }
  uint32_t target_graph_rows() const { return point_.target_graph_rows; }

 private:
  DSparkCalibrationPoint point_;
  DSparkSteadyWindowCollector collector_;
  std::vector<DSparkSteadyWindow> windows_;
};

struct DSparkCostTableEntry {
  uint32_t min_kv_length;
  uint32_t max_kv_length;
  std::vector<uint32_t> request_buckets;
  DSparkSpsCostModel model;
};

// Versioned immutable context profiles. Legacy single-table JSON remains one
// unrestricted request profile. Gaps/out-of-coverage return no table rather
// than silently applying a different context's measurements.
class DSparkCostTableBundle final {
 public:
  static std::optional<DSparkCostTableBundle> from_json(
      std::string_view contents,
      std::string_view expected_compatibility = {});

  const std::vector<DSparkCostTableEntry>& entries() const { return entries_; }
  const DSparkCostTableEntry* lookup(uint32_t requests,
                                     uint32_t max_kv_length) const;

 private:
  std::vector<DSparkCostTableEntry> entries_;
};

}  // namespace xllm
