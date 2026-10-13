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
#include <limits>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_STRIDED_MLA_KV_CACHE_REGISTRY_INC
#error "XLLM_TL_STRIDED_MLA_KV_CACHE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_STRIDED_MLA_KV_CACHE_REGISTRY_INC

int64_t get_tensor_npu_format(const torch::Tensor& tensor) {
  return at_npu::native::custom_ops::get_npu_format(tensor);
}

}  // namespace

void strided_mla_kv_cache_out(const torch::Tensor& projection,
                              const torch::Tensor& gamma,
                              const torch::Tensor& cos,
                              const torch::Tensor& sin,
                              const torch::Tensor& slots,
                              const torch::Tensor& patterns,
                              torch::Tensor& latent_cache,
                              torch::Tensor& rope_cache) {
  const std::array<const torch::Tensor*, 8> tensors = {&projection,
                                                       &gamma,
                                                       &cos,
                                                       &sin,
                                                       &slots,
                                                       &patterns,
                                                       &latent_cache,
                                                       &rope_cache};
  for (const torch::Tensor* tensor : tensors) {
    CHECK(tensor->defined());
    CHECK(tensor->device().is_privateuseone());
    CHECK_EQ(tensor->device(), projection.device());
    CHECK(tensor->is_contiguous());
    const int64_t format = get_tensor_npu_format(*tensor);
    CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW);
  }
  CHECK_EQ(projection.dim(), 2);
  CHECK_EQ(projection.size(1), 2624);
  CHECK_EQ(projection.scalar_type(), torch::kBFloat16);
  const int64_t tokens = projection.size(0);
  CHECK(tokens == 1 || tokens == 2 || tokens == 4 || tokens == 8 ||
        tokens == 16);
  CHECK_EQ(gamma.dim(), 1);
  CHECK_EQ(gamma.size(0), 512);
  CHECK_EQ(gamma.scalar_type(), torch::kBFloat16);
  CHECK_EQ(cos.dim(), 4);
  CHECK_EQ(cos.size(0), tokens);
  CHECK_EQ(cos.size(1), 1);
  CHECK_EQ(cos.size(2), 1);
  CHECK_EQ(cos.size(3), 64);
  CHECK_EQ(cos.scalar_type(), torch::kBFloat16);
  CHECK(sin.sizes() == cos.sizes());
  CHECK_EQ(sin.scalar_type(), torch::kBFloat16);
  CHECK_EQ(slots.dim(), 1);
  CHECK_EQ(slots.size(0), tokens);
  CHECK_EQ(slots.scalar_type(), torch::kInt64);
  CHECK_EQ(patterns.dim(), 2);
  CHECK_EQ(patterns.size(0), 2);
  CHECK_EQ(patterns.size(1), 64);
  CHECK_EQ(patterns.scalar_type(), torch::kInt32);
  CHECK_EQ(latent_cache.dim(), rope_cache.dim());
  CHECK(latent_cache.dim() == 3 || latent_cache.dim() == 4);
  if (latent_cache.dim() == 4) {
    CHECK_EQ(latent_cache.size(2), 1);
  }
  CHECK_EQ(latent_cache.size(-1), 512);
  CHECK_EQ(rope_cache.size(-1), 64);
  CHECK(latent_cache.sizes().slice(0, latent_cache.dim() - 1) ==
        rope_cache.sizes().slice(0, rope_cache.dim() - 1));
  CHECK_EQ(latent_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(rope_cache.scalar_type(), torch::kBFloat16);
  const int64_t cache_tokens = latent_cache.numel() / 512;
  CHECK_GT(cache_tokens, 0);
  CHECK_LE(latent_cache.numel(), std::numeric_limits<int32_t>::max());
  const std::array<const torch::Tensor*, 6> inputs = {
      &projection, &gamma, &cos, &sin, &slots, &patterns};
  for (const torch::Tensor* input : inputs) {
    CHECK(!latent_cache.is_alias_of(*input));
    CHECK(!rope_cache.is_alias_of(*input));
  }
  CHECK(!latent_cache.is_alias_of(rope_cache));
  const auto specialization = make_strided_mla_kv_cache_specialization(
      StridedMlaKvCacheNumTokens{static_cast<int32_t>(tokens)});
  const auto* entry = find_strided_mla_kv_cache_kernel_entry(specialization);
  CHECK(entry != nullptr) << available_strided_mla_kv_cache_variant_keys();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(projection.device().index()).stream();
  entry->fn(reinterpret_cast<uint8_t*>(projection.data_ptr()),
            reinterpret_cast<uint8_t*>(gamma.data_ptr()),
            reinterpret_cast<uint8_t*>(cos.data_ptr()),
            reinterpret_cast<uint8_t*>(sin.data_ptr()),
            reinterpret_cast<uint8_t*>(slots.data_ptr()),
            reinterpret_cast<uint8_t*>(latent_cache.data_ptr()),
            reinterpret_cast<uint8_t*>(rope_cache.data_ptr()),
            reinterpret_cast<uint8_t*>(patterns.data_ptr()),
            cache_tokens,
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
