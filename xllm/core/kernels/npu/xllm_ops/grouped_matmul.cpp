/* Copyright 2025-2026 The xLLM Authors.

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

#include <vector>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

std::vector<int64_t> infer_grouped_matmul_output_shape(
    const torch::Tensor& x,
    const torch::Tensor& weight) {
  CHECK(x.dim() == 2) << "x must have shape [M, K]";
  CHECK(weight.dim() == 3) << "weight must have shape [E, K, N]";
  CHECK(x.size(1) == weight.size(1))
      << "x K dimension must match weight K dimension, got " << x.size(1)
      << " vs " << weight.size(1);
  return {x.size(0), weight.size(2)};
}

}  // namespace

torch::Tensor grouped_matmul_out(const torch::Tensor& x,
                                 const torch::Tensor& weight,
                                 const torch::Tensor& scale,
                                 const torch::Tensor& per_token_scale,
                                 const torch::Tensor& group_list,
                                 int64_t split_item,
                                 int64_t group_type,
                                 int64_t group_list_type,
                                 torch::Tensor& output) {
  const auto expected_shape = infer_grouped_matmul_output_shape(x, weight);
  CHECK(output.is_contiguous()) << "output must be contiguous";
  CHECK(output.sizes().vec() == expected_shape)
      << "output shape must match grouped matmul result, got " << output.sizes()
      << " vs " << expected_shape;
  CHECK(output.device() == x.device())
      << "output and x must be on the same device";

  std::vector<torch::Tensor> x_storage{x};
  std::vector<torch::Tensor> weight_storage{weight};
  std::vector<torch::Tensor> scale_storage{scale};
  std::vector<torch::Tensor> per_token_scale_storage{per_token_scale};
  std::vector<torch::Tensor> output_storage{output};
  torch::TensorList x_list(x_storage);
  torch::TensorList weight_list(weight_storage);
  torch::TensorList scale_list(scale_storage);
  torch::TensorList per_token_scale_list(per_token_scale_storage);
  torch::TensorList output_list(output_storage);
  const c10::optional<torch::TensorList> none = c10::nullopt;
  int64_t reserved = 0;

  EXEC_NPU_CMD(aclnnGroupedMatmulV4,
               x_list,
               weight_list,
               none,
               scale_list,
               none,
               none,
               none,
               per_token_scale_list,
               group_list,
               none,
               none,
               none,
               split_item,
               group_type,
               group_list_type,
               reserved,
               output_list,
               none,
               none);
  return output;
}

}  // namespace xllm::kernel::npu
