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

"""CPU ABI/code-generation contracts for direct Indexer projection consumers."""

from __future__ import annotations

import pytest
from compiler.tilelang.targets.ascend.aot.strided_indexer_layer_norm import (
    StridedIndexerLayerNormKernel,
)

from xllm.python.kernels_npu.tilelang.strided_indexer_layer_norm import (
    STRIDED_INDEXER_TOKEN_SHAPES,
    build_strided_indexer_layer_norm_kernel,
)


def test_specializations_include_target_and_every_supported_mtp_bucket() -> None:
    assert STRIDED_INDEXER_TOKEN_SHAPES == (1, 2, 4, 8, 16)
    specs = StridedIndexerLayerNormKernel.specs()
    assert tuple(spec.specialization["num_tokens"] for spec in specs) == STRIDED_INDEXER_TOKEN_SHAPES
    assert len({spec.variant_key for spec in specs}) == len(specs)
    for spec in specs:
        spec.validate()
    assert [(field.name, field.kind) for field in StridedIndexerLayerNormKernel.dispatch_schema()] == [
        ("num_tokens", "int32")
    ]


@pytest.mark.parametrize("tokens", STRIDED_INDEXER_TOKEN_SHAPES)
def test_abi_reads_packed_projection_and_prepared_affine_inputs(tokens: int) -> None:
    kernel = build_strided_indexer_layer_norm_kernel(tokens)
    assert len(kernel.params) == 5
    buffers = [kernel.buffer_map[parameter] for parameter in kernel.params]
    assert [tuple(int(size) for size in buffer.shape) for buffer in buffers] == [
        (tokens, 160),
        (128,),
        (128,),
        (tokens, 128),
        (tokens, 32),
    ]
    assert [buffer.dtype for buffer in buffers] == ["bfloat16", "float32", "float32", "bfloat16", "bfloat16"]


@pytest.mark.parametrize("tokens", (True, False, 1.0, "1", None, 0, -1, 3, 32))
def test_unsupported_rows_are_rejected_before_lowering(tokens: object) -> None:
    with pytest.raises(ValueError, match="token shape"):
        build_strided_indexer_layer_norm_kernel(tokens)


def test_vector_codegen_has_no_cube_or_global_workspace() -> None:
    source = StridedIndexerLayerNormKernel.generate_source(4)
    assert "KERNEL_TYPE_AIV_ONLY" in source
    assert "KERNEL_TYPE_MIX_AIC_1_1" not in source
    for instruction in ("Mmad(", "CrossCoreSetFlag", "CrossCoreWaitFlag", "SetAtomicAdd", "InitGlobalMemory"):
        assert instruction not in source
