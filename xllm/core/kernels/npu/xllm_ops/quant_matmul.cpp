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

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

int64_t get_tensor_npu_format(const torch::Tensor& tensor) {
#ifdef TORCH_HIGHER_THAN_PTA6
  return at_npu::native::get_npu_format(tensor);
#else
  return at_npu::native::NPUNativeFunctions::get_npu_format(tensor);
#endif
}

std::vector<int64_t> infer_quant_matmul_output_shape(const torch::Tensor& x1,
                                                     const torch::Tensor& x2,
                                                     bool transpose2) {
  CHECK(x1.dim() >= 2) << "x1 dim must be >= 2 for quant matmul";
  CHECK(x2.dim() >= 2) << "x2 dim must be >= 2 for quant matmul";
  if (transpose2) {
    CHECK(x1.size(-1) == x2.size(-1))
        << "while transpose2 is true; x1 last dim must match x2 last dim, got "
        << x1.size(-1) << " vs " << x2.size(-1);
  } else {
    CHECK(x1.size(-1) == x2.size(-2))
        << "while transpose2 is false; x1 dim[-1] must match x2 dim[-2], got "
        << x1.size(-1) << " vs " << x2.size(-2);
  }

  auto out_shape = x1.sizes().vec();
  out_shape.back() = transpose2 ? x2.size(0) : x2.size(1);
  return out_shape;
}

torch::Tensor construct_quant_matmul_output_tensor(
    const torch::Tensor& x1,
    const torch::Tensor& x2,
    torch::ScalarType output_dtype,
    bool transpose2) {
  auto out_shape = infer_quant_matmul_output_shape(x1, x2, transpose2);
  return torch::empty(out_shape, x1.options().dtype(output_dtype));
}

void run_quant_matmul(const torch::Tensor& x1,
                      const torch::Tensor& x2,
                      bool transpose2,
                      const torch::Tensor& scale,
                      const torch::Tensor& offset,
                      const torch::Tensor& pertoken_scale,
                      const torch::Tensor& bias,
                      torch::Tensor& output) {
  const bool nz_decode_available =
      aclnn::detail::get_op_api_func_addr(
          "aclnnQuantMatmulNzDecodeGetWorkspaceSize") != nullptr &&
      aclnn::detail::get_op_api_func_addr("aclnnQuantMatmulNzDecode") !=
          nullptr;
  const bool use_nz_decode =
      nz_decode_available && !transpose2 && x1.dim() == 2 && x1.size(0) > 0 &&
      x1.size(0) <= 16 && x2.dim() == 2 && bias.defined() &&
      output.scalar_type() == torch::kBFloat16 &&
      get_tensor_npu_format(x2) == ACL_FORMAT_FRACTAL_NZ;
  if (use_nz_decode) {
    EXEC_NPU_CMD(aclnnQuantMatmulNzDecode, x1, x2, scale, bias, output);
    return;
  }
  bool transpose1 = false;
  EXEC_NPU_CMD(aclnnQuantMatmulV4,
               x1,
               x2,
               scale,
               offset,
               pertoken_scale,
               bias,
               transpose1,
               transpose2,
               output);
}

}  // namespace

torch::Tensor quant_matmul(const torch::Tensor& x1,
                           const torch::Tensor& x2,
                           const bool transpose2,
                           const torch::Tensor& scale,
                           const c10::optional<torch::Tensor>& offset,
                           const c10::optional<torch::Tensor>& pertoken_scale,
                           const c10::optional<torch::Tensor>& bias,
                           c10::optional<torch::ScalarType> output_dtype) {
  const torch::Tensor& offset_real = offset.value_or(torch::Tensor());
  const torch::Tensor& pertoken_scale_real =
      pertoken_scale.value_or(torch::Tensor());
  const torch::Tensor& bias_real = bias.value_or(torch::Tensor());
  const torch::ScalarType out_dtype = output_dtype.value_or(torch::kChar);

  torch::Tensor result =
      construct_quant_matmul_output_tensor(x1, x2, out_dtype, transpose2);
  run_quant_matmul(x1,
                   x2,
                   transpose2,
                   scale,
                   offset_real,
                   pertoken_scale_real,
                   bias_real,
                   result);
  return result;
}

torch::Tensor quant_matmul_out(
    const torch::Tensor& x1,
    const torch::Tensor& x2,
    const bool transpose2,
    const torch::Tensor& scale,
    const std::optional<torch::Tensor>& offset,
    const std::optional<torch::Tensor>& pertoken_scale,
    const std::optional<torch::Tensor>& bias,
    std::optional<torch::ScalarType> output_dtype,
    torch::Tensor& output) {
  const torch::ScalarType out_dtype = output_dtype.value_or(torch::kChar);
  const auto expected_shape =
      infer_quant_matmul_output_shape(x1, x2, transpose2);
  CHECK(output.is_contiguous()) << "output must be contiguous";
  CHECK(output.scalar_type() == out_dtype)
      << "output dtype must match output_dtype, got " << output.scalar_type()
      << " vs " << out_dtype;
  CHECK(output.sizes().vec() == expected_shape)
      << "output shape must match quant matmul result, got " << output.sizes()
      << " vs " << expected_shape;

  const torch::Tensor& offset_real = offset.value_or(torch::Tensor());
  const torch::Tensor& pertoken_scale_real =
      pertoken_scale.value_or(torch::Tensor());
  const torch::Tensor& bias_real = bias.value_or(torch::Tensor());
  run_quant_matmul(x1,
                   x2,
                   transpose2,
                   scale,
                   offset_real,
                   pertoken_scale_real,
                   bias_real,
                   output);
  return output;
}

}  // namespace xllm::kernel::npu
