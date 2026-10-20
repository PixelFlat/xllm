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

#include <algorithm>
#include <array>
#include <cstdint>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_DSPARK_GREEDY_VERIFY_COMMIT_REGISTRY_INC
#error "XLLM_TL_DSPARK_GREEDY_VERIFY_COMMIT_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {
#include XLLM_TL_DSPARK_GREEDY_VERIFY_COMMIT_REGISTRY_INC

constexpr int64_t kMaxBatchSize = 128;
constexpr int64_t kRowsPerCore = 64;

void check_vector(const torch::Tensor& tensor,
                  const torch::Device& device,
                  torch::ScalarType dtype,
                  int64_t rows) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.device(), device);
  CHECK_EQ(tensor.scalar_type(), dtype);
  CHECK_EQ(tensor.dim(), 1);
  CHECK_EQ(tensor.size(0), rows);
  CHECK(tensor.is_contiguous());
}

void check_matrix(const torch::Tensor& tensor,
                  const torch::Device& device,
                  torch::ScalarType dtype,
                  int64_t rows,
                  int64_t columns) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.device(), device);
  CHECK_EQ(tensor.scalar_type(), dtype);
  CHECK_EQ(tensor.dim(), 2);
  CHECK_EQ(tensor.size(0), rows);
  CHECK_EQ(tensor.size(1), columns);
  CHECK(tensor.is_contiguous());
}

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
  // Subtract ordered addresses instead of adding lengths to avoid pointer
  // range overflow. Tensor metadata above bounds every byte span.
  if (first_start >= second_start) {
    return first_start - second_start < second_bytes;
  }
  return second_start - first_start < first_bytes;
}

DsparkGreedyVerifyCommitSpecialization build_runtime_specialization(
    int64_t batch_size,
    int64_t capacity,
    int64_t draft_width) {
  const int64_t result_width = draft_width + 1;
  // G may include graph padding beyond (draft_width+1)*logical B. The UB target
  // buffer must cover physical G, including an idle shard with B=0.
  const int64_t required =
      std::max(batch_size, (capacity + result_width - 1) / result_width);
  int32_t bucket = 1;
  while (bucket < required) {
    bucket *= 2;
  }
  return make_dspark_greedy_verify_commit_specialization(
      DsparkGreedyVerifyCommitMaxBatchSize{bucket},
      DsparkGreedyVerifyCommitDraftWidth{static_cast<int32_t>(draft_width)});
}
}  // namespace

void dspark_greedy_verify_commit(const torch::Tensor& proposals,
                                 const torch::Tensor& target_tokens,
                                 const torch::Tensor& widths,
                                 const torch::Tensor& offsets,
                                 const torch::Tensor& owners,
                                 const torch::Tensor& columns,
                                 const torch::Tensor& source_slots,
                                 torch::Tensor& result_tokens,
                                 torch::Tensor& result_lengths,
                                 torch::Tensor& context_slots) {
  CHECK(proposals.defined());
  CHECK(proposals.device().is_privateuseone());
  CHECK_EQ(proposals.dim(), 2);
  const torch::Device device = proposals.device();
  CHECK_GE(device.index(), 0);
  const int64_t batch_size = proposals.size(0);
  const int64_t draft_width = proposals.size(1);
  CHECK(draft_width == 5 || draft_width == 7);
  const int64_t result_width = draft_width + 1;
  CHECK_LE(batch_size, kMaxBatchSize);
  check_matrix(proposals, device, torch::kInt64, batch_size, draft_width);
  CHECK(target_tokens.defined());
  CHECK_EQ(target_tokens.dim(), 1);
  const int64_t capacity = target_tokens.size(0);
  CHECK_GE(capacity, batch_size);
  CHECK_LE(capacity, kMaxBatchSize * result_width);
  check_vector(target_tokens, device, torch::kInt64, capacity);
  check_vector(widths, device, torch::kInt32, batch_size);
  check_vector(offsets, device, torch::kInt32, batch_size + 1);
  for (const torch::Tensor* tensor : {&owners, &columns, &source_slots}) {
    check_vector(*tensor, device, torch::kInt32, capacity);
  }
  check_matrix(result_tokens, device, torch::kInt64, batch_size, result_width);
  check_vector(result_lengths, device, torch::kInt32, batch_size);
  check_vector(context_slots, device, torch::kInt32, capacity);
  const std::array<const torch::Tensor*, 7> inputs = {&proposals,
                                                      &target_tokens,
                                                      &widths,
                                                      &offsets,
                                                      &owners,
                                                      &columns,
                                                      &source_slots};
  const std::array<const torch::Tensor*, 3> outputs = {
      &result_tokens, &result_lengths, &context_slots};
  for (int32_t output = 0; output < static_cast<int32_t>(outputs.size());
       ++output) {
    for (const torch::Tensor* input : inputs) {
      CHECK(!spans_overlap(*outputs[output], *input))
          << "DSpark greedy commit outputs must not overlap live inputs";
    }
    for (int32_t previous = 0; previous < output; ++previous) {
      CHECK(!spans_overlap(*outputs[output], *outputs[previous]))
          << "DSpark greedy commit output spans must be disjoint";
    }
  }
  if (capacity == 0) {
    CHECK_EQ(batch_size, 0);
    return;
  }
  // The allocator/packer guarantees widths in [1,draft_width+1], consistent
  // offsets, and owner/column bounds. These Device values never select Host
  // launches.
  const auto specialization =
      build_runtime_specialization(batch_size, capacity, draft_width);
  const auto* entry =
      find_dspark_greedy_verify_commit_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << available_dspark_greedy_verify_commit_variant_keys();
  // Each core has two 32-row vector tiles. G<=1024 bounds this to at most
  // 16 cores on the supported A2/A3 devices, without an online device query.
  const int32_t copy_cores =
      static_cast<int32_t>((capacity + kRowsPerCore - 1) / kRowsPerCore);
  const auto stream = c10_npu::getCurrentNPUStream(device.index());
  c10_npu::NPUStreamGuard stream_guard(stream);
  entry->fn(reinterpret_cast<uint8_t*>(proposals.data_ptr()),
            reinterpret_cast<uint8_t*>(target_tokens.data_ptr()),
            reinterpret_cast<uint8_t*>(widths.data_ptr()),
            reinterpret_cast<uint8_t*>(offsets.data_ptr()),
            reinterpret_cast<uint8_t*>(owners.data_ptr()),
            reinterpret_cast<uint8_t*>(columns.data_ptr()),
            reinterpret_cast<uint8_t*>(source_slots.data_ptr()),
            reinterpret_cast<uint8_t*>(result_tokens.data_ptr()),
            reinterpret_cast<uint8_t*>(result_lengths.data_ptr()),
            reinterpret_cast<uint8_t*>(context_slots.data_ptr()),
            static_cast<int32_t>(batch_size),
            static_cast<int32_t>(capacity),
            copy_cores,
            stream.stream());
}
}  // namespace xllm::kernel::npu::tilelang
