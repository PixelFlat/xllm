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

"""Decode KV cache producer geometry and runtime cache-capacity contracts."""

from __future__ import annotations

import pytest
from compiler.tilelang.targets.ascend.aot.strided_mla_kv_cache import StridedMlaKvCacheKernel

from xllm.python.kernels_npu.tilelang.strided_mla_kv_cache import (
    STRIDED_MLA_KV_TOKEN_SHAPES,
    build_strided_mla_kv_cache_kernel,
)


def test_specializations_cover_target_and_draft_token_buckets() -> None:
    assert STRIDED_MLA_KV_TOKEN_SHAPES == (1, 2, 4, 8, 16)
    specs = StridedMlaKvCacheKernel.specs()
    assert tuple(spec.specialization["num_tokens"] for spec in specs) == STRIDED_MLA_KV_TOKEN_SHAPES
    for spec in specs:
        spec.validate()


@pytest.mark.parametrize("tokens", STRIDED_MLA_KV_TOKEN_SHAPES)
def test_cache_capacity_is_a_runtime_dimension_not_a_specialization(tokens: int) -> None:
    kernel = build_strided_mla_kv_cache_kernel(tokens)
    buffers = [kernel.buffer_map[parameter] for parameter in kernel.params if parameter in kernel.buffer_map]
    assert len(buffers) == 8
    assert tuple(int(size) for size in buffers[0].shape) == (tokens, 2624)
    assert tuple(int(size) for size in buffers[1].shape) == (512,)
    assert tuple(int(size) for size in buffers[4].shape) == (tokens,)
    assert buffers[4].dtype == "int64"
    for cache, width in ((buffers[5], 512), (buffers[6], 64)):
        assert str(cache.shape[0]) == "cache_tokens"
        assert int(cache.shape[1]) == width
        assert cache.dtype == "bfloat16"


@pytest.mark.parametrize("tokens", (True, False, 1.0, "1", None, 0, -1, 3, 32))
def test_unsupported_geometry_is_rejected_before_lowering(tokens: object) -> None:
    with pytest.raises(ValueError, match="token shape"):
        build_strided_mla_kv_cache_kernel(tokens)


def test_codegen_uses_vector_only_direct_cache_writes() -> None:
    source = StridedMlaKvCacheKernel.generate_source(4)
    assert "KERNEL_TYPE_AIV_ONLY" in source
    assert "KERNEL_TYPE_MIX_AIC_1_1" not in source
    assert "cache_tokens" in source
    for instruction in ("Mmad(", "CrossCoreSetFlag", "CrossCoreWaitFlag", "SetAtomicAdd", "InitGlobalMemory"):
        assert instruction not in source
