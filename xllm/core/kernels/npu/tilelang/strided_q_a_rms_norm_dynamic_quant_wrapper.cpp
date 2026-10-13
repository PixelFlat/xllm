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

#ifndef XLLM_TL_STRIDED_Q_A_RMS_NORM_DYNAMIC_QUANT_REGISTRY_INC
#error "XLLM_TL_STRIDED_Q_A_RMS_NORM_DYNAMIC_QUANT_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {

#include XLLM_TL_STRIDED_Q_A_RMS_NORM_DYNAMIC_QUANT_REGISTRY_INC

constexpr int64_t kPackedWidth = 2624;
constexpr int64_t kQueryWidth = 2048;

}  // namespace

void strided_q_a_rms_norm_dynamic_quant_out(const torch::Tensor& projection,
                                            const torch::Tensor& gamma,
                                            torch::Tensor& output,
                                            torch::Tensor& scales) {
  const std::array<const torch::Tensor*, 4> tensors = {
      &projection, &gamma, &output, &scales};
  for (const torch::Tensor* tensor : tensors) {
    CHECK(tensor->defined());
    CHECK(tensor->device().is_privateuseone());
    CHECK_EQ(tensor->device(), projection.device());
    CHECK(tensor->is_contiguous());
  }
  CHECK_EQ(projection.dim(), 2);
  CHECK_EQ(projection.size(1), kPackedWidth);
  CHECK_EQ(projection.scalar_type(), torch::kBFloat16);
  CHECK_EQ(gamma.dim(), 1);
  CHECK_EQ(gamma.size(0), kQueryWidth);
  CHECK_EQ(gamma.scalar_type(), torch::kBFloat16);
  const int64_t num_tokens = projection.size(0);
  CHECK(num_tokens == 1 || num_tokens == 2 || num_tokens == 4 ||
        num_tokens == 8 || num_tokens == 16);
  CHECK_EQ(output.dim(), 2);
  CHECK_EQ(output.size(0), num_tokens);
  CHECK_EQ(output.size(1), kQueryWidth);
  CHECK_EQ(output.scalar_type(), torch::kInt8);
  CHECK_EQ(scales.dim(), 1);
  CHECK_EQ(scales.size(0), num_tokens);
  CHECK_EQ(scales.scalar_type(), torch::kFloat32);
  CHECK(!output.is_alias_of(projection));
  CHECK(!output.is_alias_of(gamma));
  CHECK(!output.is_alias_of(scales));
  CHECK(!scales.is_alias_of(projection));
  CHECK(!scales.is_alias_of(gamma));
  const auto specialization =
      make_strided_q_a_rms_norm_dynamic_quant_specialization(
          StridedQARmsNormDynamicQuantNumTokens{
              static_cast<int32_t>(num_tokens)});
  const auto* entry =
      find_strided_q_a_rms_norm_dynamic_quant_kernel_entry(specialization);
  CHECK(entry != nullptr)
      << available_strided_q_a_rms_norm_dynamic_quant_variant_keys();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(projection.device().index()).stream();
  entry->fn(reinterpret_cast<uint8_t*>(projection.data_ptr()),
            reinterpret_cast<uint8_t*>(gamma.data_ptr()),
            reinterpret_cast<uint8_t*>(output.data_ptr()),
            reinterpret_cast<uint8_t*>(scales.data_ptr()),
            stream);
}

}  // namespace xllm::kernel::npu::tilelang
