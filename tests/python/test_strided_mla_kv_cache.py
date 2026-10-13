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

"""Direct MLA cache contracts and changed-slot graph replay against native KV."""

from __future__ import annotations

import pytest
import torch
import torch_npu

from xllm.python.kernels_npu import _custom_op


def _arguments(tokens: int = 4, blocks: int = 3, device: str = "meta") -> list[torch.Tensor]:
    return [
        torch.empty(tokens, 2624, dtype=torch.bfloat16, device=device),
        torch.empty(512, dtype=torch.bfloat16, device=device),
        torch.empty(tokens, 1, 1, 64, dtype=torch.bfloat16, device=device),
        torch.empty(tokens, 1, 1, 64, dtype=torch.bfloat16, device=device),
        torch.empty(tokens, dtype=torch.int64, device=device),
        torch.empty(2, 64, dtype=torch.int32, device=device),
        torch.empty(blocks, 128, 512, dtype=torch.bfloat16, device=device),
        torch.empty(blocks, 128, 64, dtype=torch.bfloat16, device=device),
    ]


@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("blocks", (1, 3, 17))
@pytest.mark.parametrize("explicit_head", (False, True))
def test_fake_accepts_variable_pa_cache_capacity(tokens: int, blocks: int, explicit_head: bool) -> None:
    arguments = _arguments(tokens, blocks)
    if explicit_head:
        arguments[6] = arguments[6].unsqueeze(2)
        arguments[7] = arguments[7].unsqueeze(2)
    assert _custom_op._strided_mla_kv_cache_out_fake(*arguments) is None


@pytest.mark.parametrize("invalid", ("tokens", "projection", "slots", "patterns", "cache", "capacity", "head", "alias"))
def test_fake_rejects_unsupported_geometry_before_execution(invalid: str) -> None:
    arguments = _arguments()
    if invalid == "tokens":
        arguments = _arguments(3)
    elif invalid == "projection":
        arguments[0] = arguments[0].float()
    elif invalid == "slots":
        arguments[4] = arguments[4].int()
    elif invalid == "patterns":
        arguments[5] = arguments[5].long()
    elif invalid == "cache":
        arguments[7] = arguments[7].float()
    elif invalid == "capacity":
        arguments[6] = torch.empty(2**22, 1, 512, dtype=torch.bfloat16, device="meta")
        arguments[7] = torch.empty(2**22, 1, 64, dtype=torch.bfloat16, device="meta")
    elif invalid == "head":
        arguments[6] = arguments[6].view(1, 128, 3, 512)
        arguments[7] = arguments[7].view(1, 128, 3, 64)
    elif invalid == "alias":
        arguments[7] = arguments[6].view(-1)[: 3 * 128 * 64].view(3, 128, 64)
    with pytest.raises(ValueError):
        _custom_op._strided_mla_kv_cache_out_fake(*arguments)


@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
def test_npu_graph_replay_preserves_complete_cache_and_skips_negative_slots(tokens: int) -> None:
    torch.npu.set_device(0)
    projection, gamma, cos, sin, slots, patterns, latent, rope = _arguments(tokens, device="npu:0")
    generator = torch.Generator().manual_seed(2624512 + tokens)
    packed = torch.randn(tokens, 2624, generator=generator).bfloat16()
    affine = torch.randn(512, generator=generator).bfloat16()
    angles = torch.randn(tokens, 1, 1, 64, generator=generator)
    even = torch.arange(0, 64, 2, dtype=torch.int32)
    odd = even + 1
    patterns.copy_(torch.stack((torch.cat((even, odd)), torch.cat((odd, even)))) * 4)
    projection.copy_(packed)
    gamma.copy_(affine)
    cos.copy_(angles.cos())
    sin.copy_(angles.sin())
    slots.copy_(torch.arange(tokens, dtype=torch.int64))
    latent.fill_(3.25)
    rope.fill_(-2.5)
    wanted_latent = latent.clone()
    wanted_rope = rope.clone()
    operation = torch.ops.xllm_ops.strided_mla_kv_cache_out
    for _ in range(3):
        operation(projection, gamma, cos, sin, slots, patterns, latent, rope)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        operation(projection, gamma, cos, sin, slots, patterns, latent, rope)
    pointers = latent.data_ptr(), rope.data_ptr()
    for update, factor in enumerate((1.0, -0.5, 2.0)):
        changed = (packed.float() * factor).bfloat16()
        projection.copy_(changed)
        gamma.copy_(affine.roll(update))
        cos.copy_((angles * factor).cos())
        sin.copy_((angles * factor).sin())
        updated_slots = torch.arange(tokens, dtype=torch.int64) + 127 + update * 32
        if update == 2:
            updated_slots[-1] = -1
        slots.copy_(updated_slots)
        latent.fill_(3.25)
        rope.fill_(-2.5)
        wanted_latent.fill_(3.25)
        wanted_rope.fill_(-2.5)
        torch_npu.npu_kv_rmsnorm_rope_cache(
            projection[:, :576].view(tokens, 1, 1, 576),
            gamma,
            cos,
            sin,
            slots,
            wanted_rope,
            wanted_latent,
            epsilon=1e-5,
            cache_mode="PA",
        )
        graph.replay()
        torch.npu.synchronize()
        assert (latent.data_ptr(), rope.data_ptr()) == pointers
        assert torch.equal(projection.cpu(), changed)
        assert torch.equal(latent.cpu(), wanted_latent.cpu())
        assert torch.equal(rope.cpu(), wanted_rope.cpu())
    # Check the original int64 value before narrowing a valid cache offset.
    # In particular, -2**32 must not truncate to zero and overwrite slot zero.
    for padding in (-1, -(2**31), -(2**32), -(2**63)):
        slots.fill_(padding)
        latent.fill_(3.25)
        rope.fill_(-2.5)
        wanted_latent.fill_(3.25)
        wanted_rope.fill_(-2.5)
        graph.replay()
        torch.npu.synchronize()
        assert torch.equal(latent.cpu(), wanted_latent.cpu())
        assert torch.equal(rope.cpu(), wanted_rope.cpu())
    graph.reset()
