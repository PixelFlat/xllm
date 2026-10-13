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

"""Device-independent ABI and code-generation contracts for decode Q-A."""

from __future__ import annotations

import pytest
from compiler.tilelang.targets.ascend.aot.strided_q_a_rms_norm_dynamic_quant import (
    StridedQaRmsNormDynamicQuantKernel,
)

from xllm.python.kernels_npu.tilelang.strided_q_a_rms_norm_dynamic_quant import (
    STRIDED_Q_A_TOKEN_SHAPES,
    build_strided_q_a_rms_norm_dynamic_quant_kernel,
)


def test_specializations_include_target_and_every_supported_mtp_bucket() -> None:
    assert STRIDED_Q_A_TOKEN_SHAPES == (1, 2, 4, 8, 16)
    specs = StridedQaRmsNormDynamicQuantKernel.specs()
    assert tuple(spec.specialization["num_tokens"] for spec in specs) == STRIDED_Q_A_TOKEN_SHAPES
    assert len({spec.variant_key for spec in specs}) == len(specs)
    for spec in specs:
        spec.validate()
    assert [(field.name, field.kind) for field in StridedQaRmsNormDynamicQuantKernel.dispatch_schema()] == [
        ("num_tokens", "int32")
    ]


@pytest.mark.parametrize("tokens", STRIDED_Q_A_TOKEN_SHAPES)
def test_abi_reads_packed_projection_and_writes_only_supplied_outputs(tokens: int) -> None:
    kernel = build_strided_q_a_rms_norm_dynamic_quant_kernel(tokens)
    assert len(kernel.params) == 4
    buffers = [kernel.buffer_map[parameter] for parameter in kernel.params]
    assert [tuple(int(size) for size in buffer.shape) for buffer in buffers] == [
        (tokens, 2624),
        (2048,),
        (tokens, 2048),
        (tokens,),
    ]
    assert [buffer.dtype for buffer in buffers] == ["bfloat16", "bfloat16", "int8", "float32"]


@pytest.mark.parametrize("tokens", (True, False, 1.0, "1", None, 0, -1, 3, 32))
def test_unsupported_rows_are_rejected_before_lowering(tokens: object) -> None:
    with pytest.raises(ValueError, match="token shape"):
        build_strided_q_a_rms_norm_dynamic_quant_kernel(tokens)


def test_vector_codegen_retains_native_scalar_math_and_has_no_hidden_workspace() -> None:
    source = StridedQaRmsNormDynamicQuantKernel.generate_source(4)
    assert "KERNEL_TYPE_AIV_ONLY" in source
    assert "KERNEL_TYPE_MIX_AIC_1_1" not in source
    for helper in ("glm5_q_a_inverse_rms", "glm5_q_a_quant_multiplier", "glm5_q_a_output_scale"):
        assert source.count(helper + "(") == 2
    for instruction in ("Mmad(", "CrossCoreSetFlag", "CrossCoreWaitFlag", "AscendC::Sqrt(", "AscendC::Div("):
        assert instruction not in source
