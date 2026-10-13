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


"""Copy existing angle-table rows directly into all four RoPE layouts."""

import tilelang.language as T
from tilelang import tvm

ROPE_TABLE_GATHER_ROWS = 1048576
ROPE_TABLE_GATHER_TOKEN_SHAPES = (1, 2, 4, 8, 16)
ROPE_TABLE_GATHER_PASS_CONFIGS = {
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


def build_rope_table_gather_kernel(num_tokens: int, position_bits: int) -> tvm.tir.PrimFunc:
    if num_tokens not in ROPE_TABLE_GATHER_TOKEN_SHAPES or position_bits not in (32, 64):
        raise ValueError("RoPE gather requires T=1/2/4/8/16 and INT32/INT64 positions")
    position_dtype = f"int{position_bits}"

    @T.prim_func
    def rope_table_gather(
        cache: T.Tensor((ROPE_TABLE_GATHER_ROWS, 64), "bfloat16"),
        positions: T.Tensor((num_tokens,), position_dtype),
        half_cos: T.Tensor((num_tokens, 32), "bfloat16"),
        half_sin: T.Tensor((num_tokens, 32), "bfloat16"),
        cos: T.Tensor((num_tokens, 64), "bfloat16"),
        sin: T.Tensor((num_tokens, 64), "bfloat16"),
    ):
        with T.Kernel(num_tokens, threads=1, is_npu=True) as row:
            position = T.alloc_var("int32")
            # Scheduling owns valid positions; preserve negative-row indexing too.
            position = T.Cast("int32", positions[row])
            if position < 0:
                position = position + ROPE_TABLE_GATHER_ROWS
            cosine = T.alloc_ub((1, 32), "bfloat16")
            sine = T.alloc_ub((1, 32), "bfloat16")
            T.copy(cache[position, 0], cosine[0, :])
            T.copy(cache[position, 32], sine[0, :])
            T.copy(cosine[0, :], half_cos[row, :])
            T.copy(sine[0, :], half_sin[row, :])
            T.copy(cosine[0, :], cos[row, 0])
            T.copy(cosine[0, :], cos[row, 32])
            T.copy(sine[0, :], sin[row, 0])
            T.copy(sine[0, :], sin[row, 32])

    return rope_table_gather
