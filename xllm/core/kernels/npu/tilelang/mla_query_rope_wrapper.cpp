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

#ifndef XLLM_TL_MLA_QUERY_ROPE_REGISTRY_INC
#error "XLLM_TL_MLA_QUERY_ROPE_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_MLA_QUERY_ROPE_REGISTRY_INC

}  // namespace

torch::Tensor mla_query_rope_out(const torch::Tensor& query,
                                 const torch::Tensor& cos,
                                 const torch::Tensor& sin,
                                 const torch::Tensor& patterns,
                                 torch::Tensor& output) {
  CHECK(query.device().type() == torch::DeviceType::PrivateUse1);
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(query.size(1), 4);
  CHECK_EQ(query.size(2), 256);
  CHECK(query.scalar_type() == torch::kBFloat16);
  CHECK(patterns.scalar_type() == torch::kInt32);
  CHECK_EQ(patterns.dim(), 2);
  CHECK_EQ(patterns.size(0), 2);
  CHECK_EQ(patterns.size(1), 64);
  CHECK_EQ(output.dim(), 3);
  CHECK_EQ(output.size(0), query.size(0));
  CHECK_EQ(output.size(1), 4);
  CHECK_EQ(output.size(2), 64);
  const std::array<const torch::Tensor*, 5> tensors = {
      &query, &cos, &sin, &patterns, &output};
  for (const torch::Tensor* tensor : tensors) {
    CHECK(tensor->device() == query.device());
    CHECK(tensor->is_contiguous());
  }
  CHECK(cos.scalar_type() == query.scalar_type());
  CHECK(sin.scalar_type() == query.scalar_type());
  CHECK(output.scalar_type() == query.scalar_type());
  CHECK_EQ(cos.dim(), 4);
  CHECK_EQ(cos.size(0), query.size(0));
  CHECK_EQ(cos.size(1), 1);
  CHECK_EQ(cos.size(2), 1);
  CHECK_EQ(cos.size(3), 64);
  CHECK(sin.sizes() == cos.sizes());
  const int64_t num_tokens = query.size(0);
  CHECK(num_tokens == 1 || num_tokens == 4 || num_tokens == 16);
  const auto spec = make_mla_query_rope_specialization(
      MlaQueryRopeNumTokens{static_cast<int32_t>(num_tokens)});
  const auto* entry = find_mla_query_rope_kernel_entry(spec);
  CHECK(entry != nullptr) << available_mla_query_rope_variant_keys();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(query.device().index()).stream();
  entry->fn(reinterpret_cast<uint8_t*>(query.data_ptr()),
            reinterpret_cast<uint8_t*>(cos.data_ptr()),
            reinterpret_cast<uint8_t*>(sin.data_ptr()),
            reinterpret_cast<uint8_t*>(patterns.data_ptr()),
            reinterpret_cast<uint8_t*>(output.data_ptr()),
            stream);
  return output;
}

}  // namespace xllm::kernel::npu::tilelang
