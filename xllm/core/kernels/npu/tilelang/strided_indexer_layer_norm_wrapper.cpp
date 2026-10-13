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
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <array>
#include <cstdint>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_STRIDED_INDEXER_LAYER_NORM_REGISTRY_INC
#error "XLLM_TL_STRIDED_INDEXER_LAYER_NORM_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_STRIDED_INDEXER_LAYER_NORM_REGISTRY_INC

constexpr int64_t kPackedWidth = 160;
constexpr int64_t kKeyWidth = 128;
constexpr int64_t kWeightsWidth = 32;

}  // namespace

void strided_indexer_layer_norm_out(const torch::Tensor& projection,
                                    const torch::Tensor& gamma,
                                    const torch::Tensor& beta,
                                    torch::Tensor& key,
                                    torch::Tensor& weights) {
  const std::array<const torch::Tensor*, 5> tensors = {
      &projection, &gamma, &beta, &key, &weights};
  for (const torch::Tensor* tensor : tensors) {
    CHECK(tensor->defined());
    CHECK(tensor->device().is_privateuseone());
    CHECK_EQ(tensor->device(), projection.device());
    CHECK(tensor->is_contiguous());
  }
  CHECK_EQ(projection.dim(), 2);
  CHECK_EQ(projection.size(1), kPackedWidth);
  CHECK_EQ(projection.scalar_type(), torch::kBFloat16);
  const int64_t num_tokens = projection.size(0);
  CHECK(num_tokens == 1 || num_tokens == 2 || num_tokens == 4 ||
        num_tokens == 8 || num_tokens == 16);
  const std::array<const torch::Tensor*, 2> affine = {&gamma, &beta};
  for (const torch::Tensor* tensor : affine) {
    CHECK_EQ(tensor->dim(), 1);
    CHECK_EQ(tensor->size(0), kKeyWidth);
    CHECK_EQ(tensor->scalar_type(), torch::kFloat32);
  }
  CHECK_EQ(key.dim(), 2);
  CHECK_EQ(key.size(0), num_tokens);
  CHECK_EQ(key.size(1), kKeyWidth);
  CHECK_EQ(key.scalar_type(), torch::kBFloat16);
  CHECK_EQ(weights.dim(), 2);
  CHECK_EQ(weights.size(0), num_tokens);
  CHECK_EQ(weights.size(1), kWeightsWidth);
  CHECK_EQ(weights.scalar_type(), torch::kBFloat16);
  const std::array<const torch::Tensor*, 3> inputs = {
      &projection, &gamma, &beta};
  for (const torch::Tensor* input : inputs) {
    CHECK(!key.is_alias_of(*input));
    CHECK(!weights.is_alias_of(*input));
  }
  CHECK(!key.is_alias_of(weights));
  const auto specialization = make_strided_indexer_layer_norm_specialization(
      StridedIndexerLayerNormNumTokens{static_cast<int32_t>(num_tokens)});
  const auto* entry =
      find_strided_indexer_layer_norm_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << available_strided_indexer_layer_norm_variant_keys();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(projection.device().index()).stream();
  entry->fn(reinterpret_cast<uint8_t*>(projection.data_ptr()),
            reinterpret_cast<uint8_t*>(gamma.data_ptr()),
            reinterpret_cast<uint8_t*>(beta.data_ptr()),
            reinterpret_cast<uint8_t*>(key.data_ptr()),
            reinterpret_cast<uint8_t*>(weights.data_ptr()),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
