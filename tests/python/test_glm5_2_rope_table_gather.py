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


"""Table gather keeps angle values and isolates target/draft graph outputs."""

from __future__ import annotations

import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from xllm.python import kernels
from xllm.python.kernels_npu import _custom_op
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)
from xllm.python.models import glm5_2, glm5_2_mtp


def _rotary() -> glm5_2.Glm52YarnRotaryEmbedding:
    rotary = glm5_2.Glm52YarnRotaryEmbedding(
        64, 32, 1.0, 8000000.0, 32, 1, 1.0, 1.0, dtype=torch.bfloat16, device=torch.device("cpu")
    )
    # Exercise graph ownership with a CPU numerical stand-in, not an NPU performance claim.
    rotary._decode_gather_available = True
    return rotary


def _context(state: AclGraphExecutionState) -> ForwardContext:
    return ForwardContext(
        attention_backend=None,
        device=torch.device("cpu"),
        metadata=None,
        layer_caches=[],
        execution_state=state,
    )


def _gather(
    cache: torch.Tensor,
    positions: torch.Tensor,
    half_cos: torch.Tensor,
    half_sin: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    selected = cache[positions.long()]
    half_cos.copy_(selected[:, :32])
    half_sin.copy_(selected[:, 32:])
    cos.copy_(torch.cat((half_cos, half_cos), -1).view(-1, 1, 1, 64))
    sin.copy_(torch.cat((half_sin, half_sin), -1).view(-1, 1, 1, 64))
    return half_cos, half_sin, cos, sin


@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("position_dtype", (torch.int32, torch.int64))
def test_graph_values_are_exact_without_overwriting_live_outputs(tokens: int, position_dtype: torch.dtype) -> None:
    rotary = _rotary()
    positions = (torch.arange(tokens, dtype=position_dtype) - 1).contiguous()
    state = AclGraphExecutionState({})
    with (
        patch.object(kernels, "rope_table_gather_out", side_effect=_gather) as gather,
        forward_context(_context(state)),
    ):
        first = rotary(positions)
        pointers = tuple(value.data_ptr() for value in first)
        snapshots = tuple(value.clone() for value in first)
        positions.add_(1)
        second = rotary(positions)
    reference = glm5_2.DeepseekYarnRotaryEmbedding.forward(rotary, positions)
    for value, expected in zip(second, reference, strict=True):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        assert value.is_contiguous()
    assert all(pointer != value.data_ptr() for pointer, value in zip(pointers, second, strict=True))
    for value, snapshot in zip(first, snapshots, strict=True):
        torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
    assert gather.call_count == 2
    assert not state.persistent_buffers
    assert not torch.equal(snapshots[0], second[0])


def test_graph_entry_and_rotary_owner_outputs_are_independent() -> None:
    target, draft = _rotary(), _rotary()
    shared_pool: dict[tuple[object, ...], object] = {}
    states = [AclGraphExecutionState({}, shared_persistent_buffers=shared_pool) for _ in range(2)]
    positions = torch.arange(4, dtype=torch.int64)
    outputs = []
    with patch.object(kernels, "rope_table_gather_out", side_effect=_gather):
        for state in states:
            with forward_context(_context(state)):
                outputs.extend((target(positions), draft(positions)))
    pointers = [value.data_ptr() for result in outputs for value in result]
    assert len(set(pointers)) == 16
    assert all(not state.persistent_buffers for state in states)
    assert not shared_pool


def test_consumed_angles_are_released_without_changing_owned_metadata() -> None:
    rotary = _rotary()
    positions = torch.arange(4, dtype=torch.int64)
    state = AclGraphExecutionState({("STATIC_POSITIONS",): positions})
    with patch.object(kernels, "rope_table_gather_out", new=_gather), forward_context(_context(state)):
        for _ in range(3):
            outputs = rotary(positions)
            references = tuple(weakref.ref(value) for value in outputs)
            consumed = tuple(value.clone() for value in outputs)
            del outputs
            assert all(reference() is None for reference in references)
            expected = glm5_2.DeepseekYarnRotaryEmbedding.forward(rotary, positions)
            for actual, reference in zip(consumed, expected, strict=True):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert state.persistent_buffers == {("STATIC_POSITIONS",): positions}


@pytest.mark.parametrize("kind", ("eager", "capability", "tokens", "stride"))
def test_supported_fallback_is_selected_before_execution(kind: str) -> None:
    rotary = _rotary()
    positions = torch.arange(4, dtype=torch.int64)
    if kind == "capability":
        rotary._decode_gather_available = False
    elif kind == "tokens":
        positions = torch.arange(3, dtype=torch.int64)
    elif kind == "stride":
        positions = torch.arange(8, dtype=torch.int64)[::2]
    state = AclGraphExecutionState({})
    context = _context(state)
    if kind == "eager":
        context = ForwardContext(None, torch.device("cpu"), None, [])
    with patch.object(kernels, "rope_table_gather_out") as gather, forward_context(context):
        actual = rotary(positions)
    expected = glm5_2.DeepseekYarnRotaryEmbedding.forward(rotary, positions)
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    gather.assert_not_called()
    assert not state.persistent_buffers


def test_invalid_position_dtype_preserves_native_error_before_execution() -> None:
    rotary = _rotary()
    positions = torch.arange(4, dtype=torch.int16)
    state = AclGraphExecutionState({})
    with (
        patch.object(kernels, "rope_table_gather_out") as gather,
        forward_context(_context(state)),
        pytest.raises(IndexError),
    ):
        rotary(positions)
    with pytest.raises(IndexError):
        glm5_2.DeepseekYarnRotaryEmbedding.forward(rotary, positions)
    gather.assert_not_called()
    assert not state.persistent_buffers


@pytest.mark.parametrize("model_class", (glm5_2.Glm52Model, glm5_2_mtp.Glm52MtpModel))
@pytest.mark.parametrize("cp,split,enabled", ((1, 1, True), (2, 1, False), (1, 2, False)))
def test_target_and_mtp_factories_keep_cp_and_layerwise_paths_unchanged(
    model_class: type, cp: int, split: int, enabled: bool
) -> None:
    cfg = glm5_2.Glm52Config(cp_size=cp, layerwise_split_size=split)
    with patch.object(glm5_2, "Glm52YarnRotaryEmbedding") as constructor:
        model_class._make_rotary_embedding(SimpleNamespace(), cfg, torch.bfloat16, torch.device("cpu"))
    assert constructor.call_args.kwargs["enable_decode_gather"] is enabled


def _fake_arguments(tokens: int = 4) -> list[torch.Tensor]:
    return [
        torch.empty(1048576, 64, dtype=torch.bfloat16, device="meta"),
        torch.empty(tokens, dtype=torch.int64, device="meta"),
        torch.empty(tokens, 32, dtype=torch.bfloat16, device="meta"),
        torch.empty(tokens, 32, dtype=torch.bfloat16, device="meta"),
        torch.empty(tokens, 1, 1, 64, dtype=torch.bfloat16, device="meta"),
        torch.empty(tokens, 1, 1, 64, dtype=torch.bfloat16, device="meta"),
    ]


@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
def test_fake_contract_preserves_all_supplied_output_objects(tokens: int) -> None:
    arguments = _fake_arguments(tokens)
    assert _custom_op._rope_table_gather_out_fake(*arguments) is None


@pytest.mark.parametrize(
    "invalid", ("rows", "dtype", "positions", "tokens", "shape", "stride", "input_alias", "output_alias")
)
def test_fake_contract_rejects_invalid_geometry_and_aliasing(invalid: str) -> None:
    arguments = _fake_arguments()
    if invalid == "rows":
        arguments[0] = arguments[0][:-1]
    elif invalid == "dtype":
        arguments[2] = arguments[2].float()
    elif invalid == "positions":
        arguments[1] = arguments[1].view(2, 2)
    elif invalid == "tokens":
        arguments = _fake_arguments(3)
    elif invalid == "shape":
        arguments[4] = arguments[4].view(4, 64)
    elif invalid == "stride":
        arguments[1] = torch.empty(8, dtype=torch.int64, device="meta")[::2]
    elif invalid == "input_alias":
        arguments[2] = arguments[0][:2].view(4, 32)
    elif invalid == "output_alias":
        arguments[3] = arguments[2]
    with pytest.raises(ValueError):
        _custom_op._rope_table_gather_out_fake(*arguments)
