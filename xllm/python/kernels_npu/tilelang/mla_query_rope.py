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

"""Read interleaved query channels directly from a contiguous MLA projection."""

import tilelang.language as T
from tilelang import tvm

MLA_QUERY_ROPE_PASS_CONFIGS = {
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}
MLA_QUERY_ROPE_TOKEN_SHAPES = (1, 4, 16)


def build_mla_query_rope_kernel(num_tokens: int) -> tvm.tir.PrimFunc:
    """Keep only the decode shapes with validated complete-chain timings."""
    if num_tokens not in MLA_QUERY_ROPE_TOKEN_SHAPES:
        raise ValueError("MLA query RoPE supports token counts 1, 4 and 16")
    vector_tasks = 8 if num_tokens <= 4 else 16
    rows = num_tokens * 4
    block = (rows + vector_tasks - 1) // vector_tasks

    @T.prim_func
    def mla_query_rope(
        query: T.Tensor((rows, 256), "bfloat16"),
        cos: T.Tensor((num_tokens, 64), "bfloat16"),
        sin: T.Tensor((num_tokens, 64), "bfloat16"),
        patterns: T.Tensor((2, 64), "uint32"),
        output: T.Tensor((rows, 64), "bfloat16"),
    ):
        with T.Kernel(vector_tasks, threads=1, is_npu=True) as task:
            value_mask = T.alloc_ub((1, 64), "uint32")
            rotate_mask = T.alloc_ub((1, 64), "uint32")
            T.copy(patterns[0, :], value_mask[0, :])
            T.copy(patterns[1, :], rotate_mask[0, :])
            value_bf16 = T.alloc_ub((1, 64), "bfloat16")
            cos_bf16 = T.alloc_ub((1, 64), "bfloat16")
            sin_bf16 = T.alloc_ub((1, 64), "bfloat16")
            value = T.alloc_ub((1, 64), "float32")
            raw_value = T.alloc_ub((1, 64), "float32")
            cosine = T.alloc_ub((1, 64), "float32")
            sine = T.alloc_ub((1, 64), "float32")
            rotated = T.alloc_ub((1, 64), "float32")
            result = T.alloc_ub((1, 64), "float32")
            for local in T.serial(block):
                row = task * block + local
                if row < rows:
                    T.copy(query[row, 192], value_bf16[0, :])
                    T.copy(cos[row // 4, :], cos_bf16[0, :])
                    T.copy(sin[row // 4, :], sin_bf16[0, :])
                    T.tile.cast(raw_value, value_bf16, "CAST_NONE", 64)
                    T.tile.cast(cosine, cos_bf16, "CAST_NONE", 64)
                    T.tile.cast(sine, sin_bf16, "CAST_NONE", 64)
                    T.tile.mul(sine[0, :32], sine[0, :32], -1.0)
                    T.tile.gather(value, raw_value, value_mask, 0)
                    T.tile.gather(rotated, raw_value, rotate_mask, 0)
                    T.tile.mul(value, value, cosine)
                    T.tile.mul(rotated, rotated, sine)
                    T.tile.add(result, value, rotated)
                    T.tile.cast(value_bf16, result, "CAST_RINT", 64)
                    T.copy(value_bf16[0, :], output[row, :])

    return mla_query_rope
