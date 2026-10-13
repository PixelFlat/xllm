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


"""Native table gather is bit-exact at long and negative positions in graph replay."""

from __future__ import annotations

import weakref

import pytest
import torch

from xllm.python.model_executor.forward_context import AclGraphExecutionState, ForwardContext, forward_context
from xllm.python.models.glm5_2 import Glm52YarnRotaryEmbedding


@pytest.fixture(scope="module")
def rotary() -> Glm52YarnRotaryEmbedding:
    torch.npu.set_device(0)
    torch.set_grad_enabled(False)
    result = Glm52YarnRotaryEmbedding(
        64,
        1048576,
        1.0,
        8000000.0,
        32,
        1,
        1.0,
        1.0,
        dtype=torch.bfloat16,
        device=torch.device("npu:0"),
        enable_decode_gather=True,
    )
    assert result._decode_gather_available
    return result


def _expected(cache: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, ...]:
    rows = cache[positions.cpu().long()]
    half_cos, half_sin = rows[:, :32], rows[:, 32:]
    return (
        half_cos,
        half_sin,
        torch.cat((half_cos, half_cos), -1).view(-1, 1, 1, 64),
        torch.cat((half_sin, half_sin), -1).view(-1, 1, 1, 64),
    )


@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("position_dtype", (torch.int32, torch.int64))
def test_native_rope_gather_preserves_angles_and_updates_graph(
    rotary: Glm52YarnRotaryEmbedding, tokens: int, position_dtype: torch.dtype
) -> None:
    raw_cache = rotary.cos_sin_cache.cpu().clone()
    seed = torch.tensor([0, -1, 65536, 32768, -1048576, 262144, 1048575, 1], dtype=position_dtype)
    values = seed.repeat((tokens + seed.numel() - 1) // seed.numel())[:tokens]
    positions = values.to("npu:0")
    state = AclGraphExecutionState({})
    context = ForwardContext(None, positions.device, None, [], execution_state=state)
    with forward_context(context):
        for _ in range(3):
            outputs = rotary(positions)
        del outputs
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            outputs = rotary(positions)
        pointers = tuple(value.data_ptr() for value in outputs)
        assert len(set(pointers)) == 4
        assert not state.persistent_buffers
        for shift in (0, 1, 3):
            next_positions = values.roll(shift)
            positions.copy_(next_positions)
            for output in outputs:
                output.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            assert pointers == tuple(output.data_ptr() for output in outputs)
            for output, expected in zip(outputs, _expected(raw_cache, next_positions), strict=True):
                assert output.is_contiguous()
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
            torch.testing.assert_close(positions.cpu(), next_positions, rtol=0, atol=0)
        graph.reset()
    torch.testing.assert_close(rotary.cos_sin_cache.cpu(), raw_cache, rtol=0, atol=0)


@pytest.mark.parametrize("tokens,steps", ((1, 3), (4, 1), (16, 1)))
def test_native_consumed_angles_replay_without_execution_state_ownership(
    rotary: Glm52YarnRotaryEmbedding, tokens: int, steps: int
) -> None:
    positions = [torch.arange(tokens, dtype=torch.int64, device="npu:0") + step for step in range(steps)]
    state = AclGraphExecutionState({("STATIC_POSITIONS",): tuple(positions)})
    context = ForwardContext(None, positions[0].device, None, [], execution_state=state)
    references = []

    def _consume(position: torch.Tensor) -> torch.Tensor:
        outputs = rotary(position)
        references.extend(weakref.ref(value) for value in outputs)
        return torch.cat(tuple(value.view(tokens, -1) for value in outputs), dim=-1)

    with forward_context(context):
        for _ in range(3):
            for position in positions:
                _consume(position)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        references.clear()
        with torch.npu.graph(graph):
            result = torch.stack([_consume(position) for position in positions])
        assert all(reference() is None for reference in references)
        pointer = result.data_ptr()
        assert set(state.persistent_buffers) == {("STATIC_POSITIONS",)}
        raw_cache = rotary.cos_sin_cache.cpu()
        for shift in (1, 3, 65536):
            for position in positions:
                position.add_(shift)
            expected = torch.stack(
                [
                    torch.cat(tuple(value.reshape(tokens, -1) for value in _expected(raw_cache, position)), dim=-1)
                    for position in positions
                ]
            )
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
            assert result.data_ptr() == pointer
        graph.reset()
