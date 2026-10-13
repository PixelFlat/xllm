# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Normalize and quantize Q-A directly from the combined KV/Q-A projection."""

import tilelang.language as T
from tilelang import tvm

STRIDED_Q_A_TOKEN_SHAPES = (1, 2, 4, 8, 16)
STRIDED_Q_A_PASS_CONFIGS = {
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}
_SCALAR_SOURCE = """
#include "kernel_operator.h"
#include <cmath>

__aicore__ inline float glm5_q_a_inverse_rms(float square_sum) {
  return 1.0f / sqrt(square_sum * (1.0f / 2048.0f) + 1.0e-5f);
}
__aicore__ inline float glm5_q_a_quant_multiplier(float maximum) {
  return 127.0f / maximum;
}
__aicore__ inline float glm5_q_a_output_scale(float multiplier) {
  return 1.0f / multiplier;
}
"""


def build_strided_q_a_rms_norm_dynamic_quant_kernel(num_tokens: int) -> tvm.tir.PrimFunc:
    if type(num_tokens) is not int or num_tokens not in STRIDED_Q_A_TOKEN_SHAPES:
        raise ValueError("Unsupported decode Q-A token shape")

    @T.prim_func
    def strided_q_a_rms_norm_dynamic_quant(
        value: T.Tensor((num_tokens, 2624), "bfloat16"),
        gamma: T.Tensor((2048,), "bfloat16"),
        output: T.Tensor((num_tokens, 2048), "int8"),
        scales: T.Tensor((num_tokens,), "float32"),
    ):
        with T.block("root"):
            T.import_source(_SCALAR_SOURCE)
            with T.Kernel(num_tokens, threads=1, is_npu=True) as row:
                value_bf16 = T.alloc_ub((1, 2048), "bfloat16")
                gamma_bf16 = T.alloc_ub((1, 2048), "bfloat16")
                value_fp32 = T.alloc_ub((1, 2048), "float32")
                gamma_fp32 = T.alloc_ub((1, 2048), "float32")
                square = T.alloc_ub((1, 2048), "float32")
                sum_1024 = T.alloc_ub((1, 1024), "float32")
                tail_1024 = T.alloc_ub((1, 1024), "float32")
                sum_512 = T.alloc_ub((1, 512), "float32")
                tail_512 = T.alloc_ub((1, 512), "float32")
                sum_256 = T.alloc_ub((1, 256), "float32")
                tail_256 = T.alloc_ub((1, 256), "float32")
                sum_128 = T.alloc_ub((1, 128), "float32")
                tail_128 = T.alloc_ub((1, 128), "float32")
                sum_64 = T.alloc_ub((1, 64), "float32")
                tail_64 = T.alloc_ub((1, 64), "float32")
                reduced = T.alloc_ub((1,), "float32")
                normalized = T.alloc_ub((1, 2048), "float32")
                absolute = T.alloc_ub((1, 2048), "float32")
                maximum = T.alloc_ub((1,), "float32")
                output_scale = T.alloc_ub((1,), "float32")
                quantized_i32 = T.alloc_ub((1, 2048), "int32")
                quantized_fp32 = T.alloc_ub((1, 2048), "float32")
                quantized_fp16 = T.alloc_ub((1, 2048), "float16")
                quantized_i8 = T.alloc_ub((1, 2048), "int8")

                T.copy(value[row, 576], value_bf16[0, :])
                T.copy(gamma[:], gamma_bf16[0, :])
                T.tile.cast(value_fp32, value_bf16, "CAST_NONE", 2048)
                T.tile.cast(gamma_fp32, gamma_bf16, "CAST_NONE", 2048)
                T.tile.mul(square, value_fp32, value_fp32)
                # Preserve the native half-interval reduction tree and rounding.
                T.copy(square[0, :1024], sum_1024[0, :])
                T.copy(square[0, 1024:2048], tail_1024[0, :])
                T.tile.add(sum_1024, sum_1024, tail_1024)
                T.copy(sum_1024[0, :512], sum_512[0, :])
                T.copy(sum_1024[0, 512:1024], tail_512[0, :])
                T.tile.add(sum_512, sum_512, tail_512)
                T.copy(sum_512[0, :256], sum_256[0, :])
                T.copy(sum_512[0, 256:512], tail_256[0, :])
                T.tile.add(sum_256, sum_256, tail_256)
                T.copy(sum_256[0, :128], sum_128[0, :])
                T.copy(sum_256[0, 128:256], tail_128[0, :])
                T.tile.add(sum_128, sum_128, tail_128)
                T.copy(sum_128[0, :64], sum_64[0, :])
                T.copy(sum_128[0, 64:128], tail_64[0, :])
                T.tile.add(sum_64, sum_64, tail_64)
                T.reduce_sum(sum_64, reduced, dim=-1)
                T.set_flag("v", "s", 5)
                T.wait_flag("v", "s", 5)
                reduced[0] = T.call_pure_extern("float32", "glm5_q_a_inverse_rms", reduced[0])
                T.set_flag("s", "v", 5)
                T.wait_flag("s", "v", 5)
                T.tile.mul(normalized, value_fp32, reduced[0])
                T.tile.mul(normalized, normalized, gamma_fp32)
                T.tile.abs(absolute, normalized)
                T.reduce_max(absolute, maximum, dim=-1)
                T.set_flag("v", "s", 6)
                T.wait_flag("v", "s", 6)
                maximum[0] = T.call_pure_extern("float32", "glm5_q_a_quant_multiplier", maximum[0])
                T.set_flag("s", "v", 6)
                T.wait_flag("s", "v", 6)
                T.tile.mul(normalized, normalized, maximum[0])
                T.tile.cast(quantized_i32, normalized, "CAST_RINT", 2048)
                T.tile.cast(quantized_fp32, quantized_i32, "CAST_NONE", 2048)
                T.tile.cast(quantized_fp16, quantized_fp32, "CAST_NONE", 2048)
                T.tile.cast(quantized_i8, quantized_fp16, "CAST_TRUNC", 2048)
                T.copy(quantized_i8[0, :], output[row, :])
                output_scale[0] = T.call_pure_extern("float32", "glm5_q_a_output_scale", maximum[0])
                T.set_flag("s", "mte3", 7)
                T.wait_flag("s", "mte3", 7)
                T.copy(output_scale[:], scales[row])

    return strided_q_a_rms_norm_dynamic_quant
