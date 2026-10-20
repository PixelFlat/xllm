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
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_DSPARK_ALLOCATE_VERIFY_PREFIXES_REGISTRY_INC
#error "XLLM_TL_DSPARK_ALLOCATE_VERIFY_PREFIXES_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {
#include XLLM_TL_DSPARK_ALLOCATE_VERIFY_PREFIXES_REGISTRY_INC

constexpr int64_t kMaxBatchSize = 128;
constexpr int64_t kDraftTokens = 7;

bool spans_overlap(const torch::Tensor& first, const torch::Tensor& second) {
  if (first.numel() == 0 || second.numel() == 0) {
    return false;
  }
  const uintptr_t first_start = reinterpret_cast<uintptr_t>(first.data_ptr());
  const uintptr_t second_start = reinterpret_cast<uintptr_t>(second.data_ptr());
  const uintptr_t first_bytes =
      static_cast<uintptr_t>(first.numel() * first.element_size());
  const uintptr_t second_bytes =
      static_cast<uintptr_t>(second.numel() * second.element_size());
  return first_start < second_start + second_bytes &&
         second_start < first_start + first_bytes;
}

DsparkAllocateVerifyPrefixesSpecialization build_runtime_specialization(
    int64_t batch_size,
    bool device_control) {
  int32_t bucket = 1;
  while (bucket < batch_size) {
    bucket *= 2;
  }
  return make_dspark_allocate_verify_prefixes_specialization(
      DsparkAllocateVerifyPrefixesMaxBatchSize{bucket},
      DsparkAllocateVerifyPrefixesDeviceControl{device_control ? int32_t{1}
                                                               : int32_t{0}});
}
}  // namespace

void dspark_allocate_verify_prefixes(const torch::Tensor& confidence,
                                     torch::Tensor& widths,
                                     int64_t budget) {
  CHECK(confidence.defined());
  CHECK(widths.defined());
  CHECK(confidence.device().is_privateuseone());
  CHECK_EQ(confidence.scalar_type(), torch::kFloat32);
  CHECK_EQ(confidence.dim(), 2);
  CHECK(confidence.is_contiguous());
  const int64_t batch_size = confidence.size(0);
  CHECK_LE(batch_size, kMaxBatchSize);
  CHECK_EQ(confidence.size(1), kDraftTokens);
  CHECK_GE(budget, 0);
  CHECK_LE(budget, batch_size * kDraftTokens);
  CHECK_EQ(widths.device(), confidence.device());
  CHECK_EQ(widths.scalar_type(), torch::kInt32);
  CHECK_EQ(widths.dim(), 1);
  CHECK_EQ(widths.size(0), batch_size);
  CHECK(widths.is_contiguous());
  CHECK(!spans_overlap(confidence, widths))
      << "DSpark confidence and output widths must not overlap";
  if (batch_size == 0) {
    return;
  }
  const auto specialization =
      build_runtime_specialization(batch_size, /*device_control=*/false);
  const auto* entry =
      find_dspark_allocate_verify_prefixes_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << available_dspark_allocate_verify_prefixes_variant_keys();
  const auto stream = c10_npu::getCurrentNPUStream(confidence.device().index());
  c10_npu::NPUStreamGuard stream_guard(stream);
  entry->fn(reinterpret_cast<uint8_t*>(confidence.data_ptr()),
            reinterpret_cast<uint8_t*>(widths.data_ptr()),
            nullptr,
            static_cast<int32_t>(batch_size),
            static_cast<int32_t>(budget),
            stream.stream());
}

void dspark_allocate_verify_prefixes_graph(torch::Tensor& arena,
                                           const torch::Tensor& control,
                                           int64_t capacity) {
  CHECK(arena.defined());
  CHECK(arena.device().is_privateuseone());
  CHECK_EQ(arena.scalar_type(), torch::kUInt8);
  CHECK_EQ(arena.dim(), 1);
  CHECK(arena.is_contiguous());
  CHECK_GT(arena.numel(), 0);
  CHECK_EQ(arena.numel() % 88, 0);
  const int64_t batch_capacity = arena.numel() / 88;
  CHECK_LE(batch_capacity, kMaxBatchSize);
  CHECK_GT(capacity, 0);
  CHECK_LE(capacity, kMaxBatchSize * (kDraftTokens + 1));
  CHECK(control.defined());
  CHECK_EQ(control.device(), arena.device());
  CHECK_EQ(control.scalar_type(), torch::kInt64);
  CHECK_EQ(control.dim(), 1);
  CHECK_EQ(control.numel(), 1);
  CHECK(control.is_contiguous());
  CHECK(!spans_overlap(arena, control));
  // Prepare validates the active count before writing the control word. Read
  // it only on Device: graph replay must not turn it into a Host scalar.
  const auto specialization =
      build_runtime_specialization(batch_capacity, /*device_control=*/true);
  const auto* entry =
      find_dspark_allocate_verify_prefixes_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << available_dspark_allocate_verify_prefixes_variant_keys();
  const auto stream = c10_npu::getCurrentNPUStream(arena.device().index());
  c10_npu::NPUStreamGuard stream_guard(stream);
  entry->fn(reinterpret_cast<uint8_t*>(arena.data_ptr()),
            reinterpret_cast<uint8_t*>(arena.data_ptr()),
            reinterpret_cast<uint8_t*>(control.data_ptr()),
            static_cast<int32_t>(batch_capacity),
            static_cast<int32_t>(capacity),
            stream.stream());
}
}  // namespace xllm::kernel::npu::tilelang
