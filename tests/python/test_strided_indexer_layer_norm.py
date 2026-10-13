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

"""Direct Indexer normalization contracts and changed-input graph replay."""

from __future__ import annotations

import struct

import pytest
import torch
import torch.nn.functional as F

from xllm.python.kernels_npu import _custom_op, normalization


def _arguments(tokens: int = 4, device: str = "meta") -> list[torch.Tensor]:
    return [
        torch.empty(tokens, 160, dtype=torch.bfloat16, device=device),
        torch.empty(128, dtype=torch.float32, device=device),
        torch.empty(128, dtype=torch.float32, device=device),
        torch.empty(tokens, 128, dtype=torch.bfloat16, device=device),
        torch.empty(tokens, 32, dtype=torch.bfloat16, device=device),
    ]


@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
def test_fake_contract_retains_supplied_outputs(tokens: int) -> None:
    arguments = _arguments(tokens)
    identities = tuple(id(value) for value in arguments)
    assert _custom_op._strided_indexer_layer_norm_out_fake(*arguments) is None
    assert tuple(id(value) for value in arguments) == identities


@pytest.mark.parametrize("epsilon", (1e-6, struct.unpack("f", struct.pack("f", 1e-6))[0]))
def test_capability_accepts_literal_and_native_float_epsilon(monkeypatch: pytest.MonkeyPatch, epsilon: float) -> None:
    monkeypatch.setattr(normalization, "_STRIDED_INDEXER_LAYER_NORM_OUT", object())
    assert normalization.supports_strided_indexer_layer_norm_out(epsilon)
    monkeypatch.setattr(normalization, "_STRIDED_INDEXER_LAYER_NORM_OUT", None)
    assert not normalization.supports_strided_indexer_layer_norm_out(epsilon)


@pytest.mark.parametrize("epsilon", (0.0, 1e-5, 1.000001e-6, float("nan"), float("inf")))
def test_capability_rejects_other_epsilon_values(monkeypatch: pytest.MonkeyPatch, epsilon: float) -> None:
    monkeypatch.setattr(normalization, "_STRIDED_INDEXER_LAYER_NORM_OUT", object())
    assert not normalization.supports_strided_indexer_layer_norm_out(epsilon)


@pytest.mark.parametrize(
    "invalid", ("tokens", "projection", "gamma", "beta", "key", "weights", "stride", "input_alias", "output_alias")
)
def test_fake_contract_rejects_unsupported_geometry_and_aliases(invalid: str) -> None:
    arguments = _arguments()
    if invalid == "tokens":
        arguments = _arguments(3)
    elif invalid == "projection":
        arguments[0] = arguments[0].float()
    elif invalid == "gamma":
        arguments[1] = arguments[1].bfloat16()
    elif invalid == "beta":
        arguments[2] = arguments[2][:-1]
    elif invalid == "key":
        arguments[3] = arguments[3].float()
    elif invalid == "weights":
        arguments[4] = arguments[4].view(-1)
    elif invalid == "stride":
        arguments[0] = torch.empty(4, 320, dtype=torch.bfloat16, device="meta")[:, ::2]
    elif invalid == "input_alias":
        arguments[3] = arguments[0].view(-1)[:512].view(4, 128)
    elif invalid == "output_alias":
        arguments[4] = arguments[3].view(-1)[:128].view(4, 32)
    with pytest.raises(ValueError):
        _custom_op._strided_indexer_layer_norm_out_fake(*arguments)


@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
def test_npu_graph_replay_matches_layer_norm_and_retains_outputs(tokens: int) -> None:
    torch.npu.set_device(0)
    projection, gamma, beta, key, weights = _arguments(tokens, "npu:0")
    generator = torch.Generator().manual_seed(128160 + tokens)
    packed_fixture = torch.randn(tokens, 160, generator=generator).bfloat16()
    gamma_fixture = torch.randn(128, generator=generator).bfloat16().float()
    beta_fixture = torch.randn(128, generator=generator).bfloat16().float()
    projection.copy_(packed_fixture)
    gamma.copy_(gamma_fixture)
    beta.copy_(beta_fixture)
    operation = torch.ops.xllm_ops.strided_indexer_layer_norm_out
    for _ in range(3):
        operation(projection, gamma, beta, key, weights)
    torch.npu.synchronize()
    pointers = key.data_ptr(), weights.data_ptr()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        operation(projection, gamma, beta, key, weights)
    for factor in (0.0, 1e-3, 1.0, 64.0):
        changed = (packed_fixture.float() * factor).bfloat16()
        projection.copy_(changed)
        gamma.copy_(gamma_fixture.roll(int(factor)))
        beta.copy_(beta_fixture.roll(-int(factor)))
        key.fill_(float("nan"))
        weights.fill_(float("nan"))
        expected = F.layer_norm(projection[:, :128], (128,), gamma, beta, 1e-6)
        graph.replay()
        torch.npu.synchronize()
        assert (key.data_ptr(), weights.data_ptr()) == pointers
        assert torch.equal(projection.cpu(), changed)
        assert torch.equal(key.cpu(), expected.cpu())
        assert torch.equal(weights.cpu(), changed[:, 128:])
    graph.reset()
