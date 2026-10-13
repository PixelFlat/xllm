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

"""Q-A output contracts and native changed-input graph replay."""

from __future__ import annotations

import pytest
import torch

from xllm.python.kernels_npu import _custom_op


def _arguments(tokens: int = 4, device: str = "meta") -> list[torch.Tensor]:
    return [
        torch.empty(tokens, 2624, dtype=torch.bfloat16, device=device),
        torch.empty(2048, dtype=torch.bfloat16, device=device),
        torch.empty(tokens, 2048, dtype=torch.int8, device=device),
        torch.empty(tokens, dtype=torch.float32, device=device),
    ]


@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
def test_fake_contract_uses_supplied_outputs(tokens: int) -> None:
    arguments = _arguments(tokens)
    identities = tuple(id(value) for value in arguments)
    assert _custom_op._strided_q_a_rms_norm_dynamic_quant_out_fake(*arguments) is None
    assert tuple(id(value) for value in arguments) == identities


@pytest.mark.parametrize(
    "invalid", ("projection", "tokens", "dtype", "gamma", "output", "scale", "stride", "input_alias", "output_alias")
)
def test_fake_contract_rejects_unsupported_layout_and_aliases(invalid: str) -> None:
    arguments = _arguments()
    if invalid == "projection":
        arguments[0] = arguments[0].view(-1)
    elif invalid == "tokens":
        arguments = _arguments(3)
    elif invalid == "dtype":
        arguments[0] = arguments[0].float()
    elif invalid == "gamma":
        arguments[1] = arguments[1][:-1]
    elif invalid == "output":
        arguments[2] = arguments[2].to(torch.int16)
    elif invalid == "scale":
        arguments[3] = arguments[3].view(2, 2)
    elif invalid == "stride":
        arguments[0] = torch.empty(4, 5248, dtype=torch.bfloat16, device="meta")[:, ::2]
    elif invalid == "input_alias":
        arguments[2] = arguments[0].view(torch.int8).view(-1)[:8192].view(4, 2048)
    elif invalid == "output_alias":
        arguments[3] = arguments[2].view(torch.float32).view(-1)[:4]
    with pytest.raises(ValueError):
        _custom_op._strided_q_a_rms_norm_dynamic_quant_out_fake(*arguments)


@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
def test_npu_graph_replay_matches_native_quantization_and_retains_output_addresses(tokens: int) -> None:
    torch.npu.set_device(0)
    projection, gamma, output, scales = _arguments(tokens, "npu:0")
    generator = torch.Generator().manual_seed(2048576 + tokens)
    input_fixture = torch.randn(tokens, 2624, generator=generator).bfloat16()
    gamma_fixture = torch.randn(2048, generator=generator).bfloat16()
    projection.copy_(input_fixture)
    gamma.copy_(gamma_fixture)
    operation = torch.ops.xllm_ops.strided_q_a_rms_norm_dynamic_quant_out
    for _ in range(3):
        operation(projection, gamma, output, scales)
    torch.npu.synchronize()
    pointers = output.data_ptr(), scales.data_ptr()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        operation(projection, gamma, output, scales)
    for factor in (0.0, 1e-3, 1.0, 64.0):
        projection.copy_((input_fixture.float() * factor).bfloat16())
        gamma.copy_(gamma_fixture.roll(int(factor)))
        output.fill_(-128)
        scales.fill_(float("nan"))
        reference = torch.ops.xllm_ops.rms_norm_dynamic_quant(projection[:, 576:], gamma, 1e-5)
        graph.replay()
        torch.npu.synchronize()
        assert (output.data_ptr(), scales.data_ptr()) == pointers
        assert torch.equal(output.cpu(), reference[0].cpu())
        assert torch.equal(scales.cpu(), reference[1].cpu())
    graph.reset()
