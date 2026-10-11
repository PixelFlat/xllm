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

"""Numerical and replay checks for the shared-expert NZ output projection."""

from __future__ import annotations

from typing import Callable

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


def _require_quant_matmul_out() -> Callable[..., torch.Tensor]:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    from xllm import xllm_export  # noqa: F401

    op = getattr(torch.ops.xllm_ops, "quant_matmul_out", None)
    assert op is not None, "requires a build with native quant_matmul_out"
    return op


def _inputs(token_count: int) -> tuple[torch.Tensor, ...]:
    from xllm.python.kernels_npu.linear import prepare_quant_weight

    generator = torch.Generator().manual_seed(29)
    x1 = torch.randint(-8, 8, (token_count, 64), dtype=torch.int8, generator=generator).npu()
    weight = torch.randint(-8, 8, (96, 64), dtype=torch.int8, generator=generator)
    x2 = prepare_quant_weight(weight.npu())
    scale = torch.linspace(0.01, 0.05, 96).npu()
    pertoken = torch.linspace(0.5, 1.0, token_count).npu()
    output = torch.empty(token_count, 96, dtype=torch.bfloat16, device=x1.device)
    return x1, x2, scale, pertoken, output, weight


def _reference(
    x1: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    pertoken: torch.Tensor,
) -> torch.Tensor:
    accumulated = x1.cpu().to(torch.int32) @ weight.t().to(torch.int32)
    return (accumulated.float() * scale.cpu() * pertoken.cpu().unsqueeze(-1)).to(torch.bfloat16)


@pytest.mark.parametrize("token_count", [1, 4])
def test_quant_matmul_out_writes_nz_weight_result(token_count: int) -> None:
    # Four target-verification token rows still belong to a single K=3 sequence.
    op = _require_quant_matmul_out()
    x1, x2, scale, pertoken, output, weight = _inputs(token_count)
    expected = _reference(x1, weight, scale, pertoken)

    result = op(x1, x2, False, scale, None, pertoken, None, torch.bfloat16, output)
    torch.npu.synchronize()

    assert result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output.cpu(), expected, rtol=0.01, atol=0.015)


def test_quant_matmul_out_nz_graph_replay_updates_output() -> None:
    op = _require_quant_matmul_out()
    x1, x2, scale, pertoken, output, weight = _inputs(1)
    output_ptr = output.data_ptr()
    stream = torch.npu.Stream(device=x1.device)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        op(x1, x2, False, scale, None, pertoken, None, torch.bfloat16, output)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        op(x1, x2, False, scale, None, pertoken, None, torch.bfloat16, output)

    for value in (-3, 5):
        x1.fill_(value)
        graph.replay()
        torch.npu.synchronize()
        expected = _reference(x1, weight, scale, pertoken)
        assert output.data_ptr() == output_ptr
        torch.testing.assert_close(output.cpu(), expected, rtol=0.01, atol=0.015)


def _static_inputs(token_count: int, in_features: int, out_features: int) -> tuple[torch.Tensor, ...]:
    from xllm.python.kernels_npu.linear import prepare_quant_weight

    generator = torch.Generator().manual_seed(31)
    x1 = torch.randint(-2, 3, (token_count, in_features), dtype=torch.int8, generator=generator).npu()
    weight = torch.randint(-2, 3, (out_features, in_features), dtype=torch.int8, generator=generator)
    x2 = prepare_quant_weight(weight.npu())
    scale = torch.linspace(0.01, 0.05, out_features).npu()
    bias = torch.arange(out_features, dtype=torch.int32).remainder(17).npu()
    output = torch.empty(token_count, out_features, dtype=torch.bfloat16, device=x1.device)
    return x1, x2, scale, bias, output, weight


def _static_reference(x1: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    accumulated = x1.cpu().to(torch.int32) @ weight.t().to(torch.int32)
    return ((accumulated + bias.cpu()).float() * scale.cpu()).to(torch.bfloat16)


@pytest.mark.parametrize("token_count", [1, 8, 17])
@pytest.mark.parametrize("projection", [(64, 96), (6144, 2624), (5120, 1280)], ids=["small", "glm", "qwen"])
@pytest.mark.parametrize("use_out", [False, True], ids=["allocating", "out"])
def test_static_quant_matmul_respects_nz_projection_contract(
    token_count: int, projection: tuple[int, int], use_out: bool
) -> None:
    out_op = _require_quant_matmul_out()
    x1, x2, scale, bias, output, weight = _static_inputs(token_count, *projection)
    expected = _static_reference(x1, weight, scale, bias)

    if use_out:
        result = out_op(x1, x2, False, scale, None, None, bias, torch.bfloat16, output)
        assert result.data_ptr() == output.data_ptr()
    else:
        result = torch.ops.xllm_ops.quant_matmul(x1, x2, False, scale, None, None, bias, torch.bfloat16)
    torch.npu.synchronize()

    torch.testing.assert_close(result.cpu(), expected, rtol=0.01, atol=0.015)


@pytest.mark.parametrize("projection", [(6144, 2624), (5120, 1280)], ids=["glm", "qwen"])
def test_static_quant_matmul_nz_graph_replays_live_inputs(projection: tuple[int, int]) -> None:
    op = _require_quant_matmul_out()
    x1, x2, scale, bias, output, weight = _static_inputs(8, *projection)
    output_ptr = output.data_ptr()
    stream = torch.npu.Stream(device=x1.device)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        op(x1, x2, False, scale, None, None, bias, torch.bfloat16, output)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        op(x1, x2, False, scale, None, None, bias, torch.bfloat16, output)

    for value in (-1, 2):
        x1.fill_(value)
        graph.replay()
        torch.npu.synchronize()
        expected = _static_reference(x1, weight, scale, bias)
        assert output.data_ptr() == output_ptr
        torch.testing.assert_close(output.cpu(), expected, rtol=0.01, atol=0.015)
