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

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    KVLoadContext,
    forward_context,
    get_forward_context_or_none,
    wait_for_kv_layer,
)
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.glm5_2 import Glm52DecoderLayer, Glm52Model


@pytest.mark.parametrize(("group_size", "num_layers", "events"), [(0, 3, 3), (1, 0, 1), (2, 5, 2), (2, 5, 5)])
def test_invalid_load_geometry_is_rejected(group_size: int, num_layers: int, events: int) -> None:
    synchronizer = MagicMock()
    synchronizer.size.return_value = events
    with pytest.raises(ValueError):
        KVLoadContext(synchronizer, group_size, num_layers)
    synchronizer.wait_group.assert_not_called()


def test_failed_copy_stops_before_the_group_computes() -> None:
    synchronizer = MagicMock()
    synchronizer.size.return_value = 2
    synchronizer.wait_group.side_effect = [True, False]
    load = KVLoadContext(synchronizer, 2, 3)
    load.wait_layer(0)
    load.wait_layer(1)
    with pytest.raises(RuntimeError, match="reload failed for layer group 1"):
        load.wait_layer(2)
    with pytest.raises(RuntimeError, match="did not consume"):
        load.check_consumed()


def test_missing_and_out_of_order_groups_are_rejected() -> None:
    synchronizer = MagicMock()
    synchronizer.size.return_value = 3
    load = KVLoadContext(synchronizer, 2, 5)
    with pytest.raises(RuntimeError, match="layer order"):
        load.wait_layer(2)
    synchronizer.wait_group.assert_not_called()
    load.wait_layer(0)
    with pytest.raises(RuntimeError, match="did not consume"):
        load.check_consumed()


def test_per_batch_load_dependencies_cannot_enter_graph_capture() -> None:
    synchronizer = MagicMock()
    synchronizer.size.return_value = 1
    context = ForwardContext(
        MagicMock(),
        torch.device("cpu"),
        MagicMock(),
        [],
        execution_state=AclGraphExecutionState({}),
        kv_load=KVLoadContext(synchronizer, 1, 1),
    )
    with forward_context(context), pytest.raises(RuntimeError, match="cannot be captured"):
        wait_for_kv_layer(0)
    synchronizer.wait_group.assert_not_called()


class _CacheAttention(nn.Module):
    def __init__(self, index: int, cache: torch.Tensor, trace: list[tuple[str, int]]) -> None:
        super().__init__()
        self.index = index
        self.cache = cache
        self.trace = trace

    def _can_fuse_input_norm_quant(self) -> bool:
        return False

    def forward(self, hidden: torch.Tensor, *args: object) -> tuple[torch.Tensor, None]:
        assert self.cache[self.index].item() == 7 + self.index
        self.trace.append(("compute", self.index))
        return hidden + self.cache[self.index], None


class _IdentityNorm(nn.Module):
    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return hidden if residual is None else (hidden, residual)


def _make_cache_layer(index: int, cache: torch.Tensor, trace: list[tuple[str, int]]) -> Glm52DecoderLayer:
    layer = Glm52DecoderLayer.__new__(Glm52DecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_id = index
    layer.input_layernorm = _IdentityNorm()
    layer.self_attn = _CacheAttention(index, cache, trace)
    layer.post_attention_layernorm = _IdentityNorm()
    layer.mlp = nn.Identity()
    layer._fuse_dense_norm_quant = False
    return layer


@pytest.mark.parametrize(("layers", "group_size"), [(6, 1), (6, 2), (5, 2), (3, 8)])
def test_glm52_waits_before_cache_reads_and_keeps_layer_export_events(layers: int, group_size: int) -> None:
    cache = torch.arange(layers).float() + 7
    trace: list[tuple[str, int]] = []
    model = Glm52Model.__new__(Glm52Model)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(enable_attn_dp_weight_sharding=False)
    model.embed_tokens = nn.Embedding(16, 4)
    model.layers = nn.ModuleList([_make_cache_layer(index, cache, trace) for index in range(layers)])
    model.norm = _IdentityNorm()
    model.aux_hidden_capture = AuxHiddenCapture(())
    model._prepare_layer_inputs = MagicMock(
        side_effect=lambda hidden, positions, cp: (hidden, (positions,) * 4, (positions, positions))
    )
    model._prepare_mla_slots = MagicMock(return_value=None)
    tokens = torch.tensor([1, 2])
    positions = torch.tensor([0, 1])
    with forward_context(ForwardContext(MagicMock(), torch.device("cpu"), MagicMock(), [])):
        expected_output = model(tokens, positions)
    expected_trace = []
    for layer in range(layers):
        if layer % group_size == 0:
            expected_trace.append(("wait", layer // group_size))
        expected_trace.extend([("compute", layer), ("export", layer)])
    # Reuse the model across forwards; each invocation owns fresh load state.
    for _ in range(2):
        trace.clear()
        cache.zero_()
        synchronizer = MagicMock()
        synchronizer.size.return_value = (layers + group_size - 1) // group_size

        def wait_group(group: int) -> bool:
            start = group * group_size
            end = min(start + group_size, layers)
            assert torch.count_nonzero(cache[end:]).item() == 0
            cache[start:end] = torch.arange(start, end).float() + 7
            trace.append(("wait", group))
            return True

        def record_event(layer: int) -> bool:
            trace.append(("export", layer))
            return True

        synchronizer.wait_group.side_effect = wait_group
        producer = MagicMock()
        producer.record_event.side_effect = record_event
        load = KVLoadContext(synchronizer, group_size, layers)
        context = ForwardContext(
            MagicMock(), torch.device("cpu"), MagicMock(), [], layer_synchronizer=producer, kv_load=load
        )
        with forward_context(context):
            actual_output = model(tokens, positions)
        load.check_consumed()
        assert get_forward_context_or_none() is None
        wait_for_kv_layer(0)
        assert trace == expected_trace
        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
