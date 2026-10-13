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

#ifndef XLLM_TL_ROPE_TABLE_GATHER_REGISTRY_INC
#error "XLLM_TL_ROPE_TABLE_GATHER_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_ROPE_TABLE_GATHER_REGISTRY_INC

constexpr int64_t kCacheRows = 1048576;
constexpr int64_t kHeadDim = 64;

}  // namespace

void rope_table_gather_out(const torch::Tensor& cache,
                           const torch::Tensor& positions,
                           torch::Tensor& half_cos,
                           torch::Tensor& half_sin,
                           torch::Tensor& cos,
                           torch::Tensor& sin) {
  CHECK(cache.device().is_privateuseone());
  CHECK_EQ(cache.dim(), 2);
  CHECK_EQ(cache.size(0), kCacheRows);
  CHECK_EQ(cache.size(1), kHeadDim);
  CHECK_EQ(cache.scalar_type(), torch::kBFloat16);
  CHECK(cache.is_contiguous());
  CHECK_EQ(positions.dim(), 1);
  CHECK(positions.scalar_type() == torch::kInt32 ||
        positions.scalar_type() == torch::kInt64);
  CHECK_EQ(positions.device(), cache.device());
  CHECK(positions.is_contiguous());
  const int64_t num_tokens = positions.size(0);
  CHECK(num_tokens == 1 || num_tokens == 2 || num_tokens == 4 ||
        num_tokens == 8 || num_tokens == 16);
  const std::array<const torch::Tensor*, 4> outputs = {
      &half_cos, &half_sin, &cos, &sin};
  for (const torch::Tensor* output : outputs) {
    CHECK_EQ(output->scalar_type(), cache.scalar_type());
    CHECK_EQ(output->device(), cache.device());
    CHECK(output->is_contiguous());
    CHECK(!output->is_alias_of(cache));
    CHECK(!output->is_alias_of(positions));
  }
  CHECK(!half_cos.is_alias_of(half_sin));
  CHECK(!half_cos.is_alias_of(cos));
  CHECK(!half_cos.is_alias_of(sin));
  CHECK(!half_sin.is_alias_of(cos));
  CHECK(!half_sin.is_alias_of(sin));
  CHECK(!cos.is_alias_of(sin));
  CHECK_EQ(half_cos.dim(), 2);
  CHECK_EQ(half_cos.size(0), num_tokens);
  CHECK_EQ(half_cos.size(1), kHeadDim / 2);
  CHECK(half_sin.sizes() == half_cos.sizes());
  CHECK_EQ(cos.dim(), 4);
  CHECK_EQ(cos.size(0), num_tokens);
  CHECK_EQ(cos.size(1), 1);
  CHECK_EQ(cos.size(2), 1);
  CHECK_EQ(cos.size(3), kHeadDim);
  CHECK(sin.sizes() == cos.sizes());
  const auto spec = make_rope_table_gather_specialization(
      RopeTableGatherNumTokens{static_cast<int32_t>(num_tokens)},
      RopeTableGatherPositionBits{
          static_cast<int32_t>(positions.element_size() * 8)});
  const auto* entry = find_rope_table_gather_kernel_entry(spec);
  CHECK(entry != nullptr) << available_rope_table_gather_variant_keys();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(cache.device().index()).stream();
  entry->fn(reinterpret_cast<uint8_t*>(cache.data_ptr()),
            reinterpret_cast<uint8_t*>(positions.data_ptr()),
            reinterpret_cast<uint8_t*>(half_cos.data_ptr()),
            reinterpret_cast<uint8_t*>(half_sin.data_ptr()),
            reinterpret_cast<uint8_t*>(cos.data_ptr()),
            reinterpret_cast<uint8_t*>(sin.data_ptr()),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
