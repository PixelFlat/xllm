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

"""Write paged MLA caches directly from the combined KV/Q-A projection."""

import tilelang.language as T
from tilelang import tvm

STRIDED_MLA_KV_TOKEN_SHAPES = (1, 2, 4, 8, 16)
STRIDED_MLA_KV_PASS_CONFIGS = {
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


def build_strided_mla_kv_cache_kernel(num_tokens: int) -> tvm.tir.PrimFunc:
    if type(num_tokens) is not int or num_tokens not in STRIDED_MLA_KV_TOKEN_SHAPES:
        raise ValueError("Unsupported decode MLA KV token shape")
    cache_tokens = T.symbolic("cache_tokens")

    @T.prim_func
    def strided_mla_kv_cache(
        projected: T.Tensor((num_tokens, 2624), "bfloat16"),
        gamma: T.Tensor((512,), "bfloat16"),
        cosine: T.Tensor((num_tokens, 64), "bfloat16"),
        sine: T.Tensor((num_tokens, 64), "bfloat16"),
        slots: T.Tensor((num_tokens,), "int64"),
        latent_cache: T.Tensor((cache_tokens, 512), "bfloat16"),
        rope_cache: T.Tensor((cache_tokens, 64), "bfloat16"),
        patterns: T.Tensor((2, 64), "uint32"),
    ):
        with T.Kernel(num_tokens, threads=1, is_npu=True) as row:
            slot_value = T.alloc_var("int64")
            slot_value = slots[row]
            # The cache owner supplies unique, valid positive slots. Negative
            # padding slots leave the complete existing cache contents intact.
            if slot_value >= 0:
                slot = T.alloc_var("int32")
                slot = T.Cast("int32", slot_value)
                latent_bf16 = T.alloc_ub((1, 512), "bfloat16")
                gamma_bf16 = T.alloc_ub((1, 512), "bfloat16")
                latent = T.alloc_ub((1, 512), "float32")
                affine = T.alloc_ub((1, 512), "float32")
                square = T.alloc_ub((1, 512), "float32")
                sum_256 = T.alloc_ub((1, 256), "float32")
                tail_256 = T.alloc_ub((1, 256), "float32")
                sum_128 = T.alloc_ub((1, 128), "float32")
                tail_128 = T.alloc_ub((1, 128), "float32")
                sum_64 = T.alloc_ub((1, 64), "float32")
                tail_64 = T.alloc_ub((1, 64), "float32")
                reduced = T.alloc_ub((1,), "float32")
                one = T.alloc_ub((1,), "float32")
                T.copy(projected[row, 0], latent_bf16[0, :])
                T.copy(gamma[:], gamma_bf16[0, :])
                T.tile.cast(latent, latent_bf16, "CAST_NONE", 512)
                T.tile.cast(affine, gamma_bf16, "CAST_NONE", 512)
                T.tile.mul(square, latent, latent)
                T.copy(square[0, :256], sum_256[0, :])
                T.copy(square[0, 256:512], tail_256[0, :])
                T.tile.add(sum_256, sum_256, tail_256)
                T.copy(sum_256[0, :128], sum_128[0, :])
                T.copy(sum_256[0, 128:256], tail_128[0, :])
                T.tile.add(sum_128, sum_128, tail_128)
                T.copy(sum_128[0, :64], sum_64[0, :])
                T.copy(sum_128[0, 64:128], tail_64[0, :])
                T.tile.add(sum_64, sum_64, tail_64)
                T.reduce_sum(sum_64, reduced, dim=-1)
                T.tile.mul(reduced, reduced, 1.0 / 512.0)
                T.tile.add(reduced, reduced, 1e-5)
                T.tile.sqrt(reduced, reduced)
                one[0] = 1.0
                T.tile.div(reduced, one, reduced)
                T.tile.mul(latent, latent, reduced[0])
                T.tile.mul(latent, latent, affine)
                T.tile.cast(latent_bf16, latent, "CAST_RINT", 512)
                T.copy(latent_bf16[0, :], latent_cache[slot, :])

                rope_bf16 = T.alloc_ub((1, 64), "bfloat16")
                cos_bf16 = T.alloc_ub((1, 64), "bfloat16")
                sin_bf16 = T.alloc_ub((1, 64), "bfloat16")
                raw_rope = T.alloc_ub((1, 64), "float32")
                rope = T.alloc_ub((1, 64), "float32")
                rotated = T.alloc_ub((1, 64), "float32")
                cos = T.alloc_ub((1, 64), "float32")
                sin = T.alloc_ub((1, 64), "float32")
                result = T.alloc_ub((1, 64), "float32")
                value_mask = T.alloc_ub((1, 64), "uint32")
                rotate_mask = T.alloc_ub((1, 64), "uint32")
                T.copy(patterns[0, :], value_mask[0, :])
                T.copy(patterns[1, :], rotate_mask[0, :])
                T.copy(projected[row, 512], rope_bf16[0, :])
                T.copy(cosine[row, :], cos_bf16[0, :])
                T.copy(sine[row, :], sin_bf16[0, :])
                T.tile.cast(raw_rope, rope_bf16, "CAST_NONE", 64)
                T.tile.cast(cos, cos_bf16, "CAST_NONE", 64)
                T.tile.cast(sin, sin_bf16, "CAST_NONE", 64)
                T.tile.mul(sin[0, :32], sin[0, :32], -1.0)
                T.tile.gather(rope, raw_rope, value_mask, 0)
                T.tile.gather(rotated, raw_rope, rotate_mask, 0)
                T.tile.mul(rope, rope, cos)
                T.tile.mul(rotated, rotated, sin)
                T.tile.add(result, rope, rotated)
                T.tile.cast(rope_bf16, result, "CAST_RINT", 64)
                T.copy(rope_bf16[0, :], rope_cache[slot, :])

    return strided_mla_kv_cache
