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

#include <torch/torch.h>

#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

bool has_moe_grouped_matmul_swiglu_quant() {
  static const bool available =
      aclnn::detail::get_op_api_func_addr(
          "aclnnMoeGroupedMatmulSwigluQuantGetWorkspaceSize") != nullptr &&
      aclnn::detail::get_op_api_func_addr("aclnnMoeGroupedMatmulSwigluQuant") !=
          nullptr;
  return available;
}

std::tuple<torch::Tensor, torch::Tensor> moe_grouped_matmul_swiglu_quant(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& x_scale,
    const torch::Tensor& group_list) {
  CHECK(x.dim() == 2) << "x must be [M, K]";
  CHECK(weight.dim() == 3) << "weight must be [E, K, 2N]";
  CHECK(weight_scale.dim() == 2) << "weight_scale must be [E, 2N]";
  CHECK(x_scale.dim() == 1) << "x_scale must be [M]";
  CHECK(group_list.dim() == 2 && group_list.size(1) == 2)
      << "group_list must be [E, 2]";
  CHECK(weight.size(2) % 2 == 0) << "the fused projection width must be even";

  const int64_t output_width = weight.size(2) / 2;
  torch::Tensor output =
      torch::empty({x.size(0), output_width}, x.options().dtype(torch::kInt8));
  torch::Tensor output_scale =
      torch::empty({x.size(0)}, x.options().dtype(torch::kFloat32));
  EXEC_NPU_CMD(aclnnMoeGroupedMatmulSwigluQuant,
               x,
               weight,
               weight_scale,
               x_scale,
               group_list,
               output,
               output_scale);
  return std::make_tuple(output, output_scale);
}

}  // namespace xllm::kernel::npu
