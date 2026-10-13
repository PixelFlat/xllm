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

"""Read packed Indexer projections without materializing their strided views."""

import tilelang.language as T
from tilelang import tvm

STRIDED_INDEXER_TOKEN_SHAPES = (1, 2, 4, 8, 16)
STRIDED_INDEXER_PASS_CONFIGS = {
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


def build_strided_indexer_layer_norm_kernel(num_tokens: int) -> tvm.tir.PrimFunc:
    if type(num_tokens) is not int or num_tokens not in STRIDED_INDEXER_TOKEN_SHAPES:
        raise ValueError("Unsupported decode Indexer token shape")

    @T.prim_func
    def strided_indexer_layer_norm(
        projected: T.Tensor((num_tokens, 160), "bfloat16"),
        gamma: T.Tensor((128,), "float32"),
        beta: T.Tensor((128,), "float32"),
        key: T.Tensor((num_tokens, 128), "bfloat16"),
        weights: T.Tensor((num_tokens, 32), "bfloat16"),
    ):
        with T.Kernel(num_tokens, threads=1, is_npu=True) as row:
            raw = T.alloc_ub((1, 128), "bfloat16")
            value = T.alloc_ub((1, 128), "float32")
            affine_gamma = T.alloc_ub((1, 128), "float32")
            affine_beta = T.alloc_ub((1, 128), "float32")
            centered = T.alloc_ub((1, 128), "float32")
            square = T.alloc_ub((1, 128), "float32")
            mean = T.alloc_ub((1,), "float32")
            variance = T.alloc_ub((1,), "float32")
            one = T.alloc_ub((1,), "float32")
            raw_weights = T.alloc_ub((1, 32), "bfloat16")
            T.copy(projected[row, :128], raw[0, :])
            # Affine inputs are promoted once after checkpoint values have been
            # rounded to their runtime BF16 dtype, preserving native semantics.
            T.copy(gamma[:], affine_gamma[0, :])
            T.copy(beta[:], affine_beta[0, :])
            T.tile.cast(value, raw, "CAST_NONE", 128)
            T.reduce_sum(value, mean, dim=-1)
            T.tile.mul(mean, mean, 1.0 / 128.0)
            mean[0] = -mean[0]
            T.tile.add(centered, value, mean[0])
            T.tile.mul(square, centered, centered)
            T.reduce_sum(square, variance, dim=-1)
            T.tile.mul(variance, variance, 1.0 / 128.0)
            T.tile.add(variance, variance, 1e-6)
            T.tile.sqrt(variance, variance)
            one[0] = 1.0
            T.tile.div(variance, one, variance)
            T.tile.mul(centered, centered, variance[0])
            T.tile.mul(centered, centered, affine_gamma)
            T.tile.add(centered, centered, affine_beta)
            T.tile.cast(raw, centered, "CAST_RINT", 128)
            T.copy(raw[0, :], key[row, :])
            T.copy(projected[row, 128:160], raw_weights[0, :])
            T.copy(raw_weights[0, :], weights[row, :])

    return strided_indexer_layer_norm
