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

"""Direct Indexer dispatch, normalization state, and graph buffer ownership."""

from __future__ import annotations

import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

from xllm.python.model_executor.forward_context import AclGraphExecutionState, forward_context
from xllm.python.models import glm5_2


def _indexer(model_type: str = "glm_moe_dsa") -> glm5_2.Glm52Indexer:
    cfg = glm5_2.Glm52Config(
        model_type=model_type,
        hidden_size=16,
        q_lora_rank=16,
        index_n_heads=32,
        index_head_dim=128,
        qk_rope_head_dim=64,
        tp_size=16,
        index_topk=1,
        indexer_rope_interleave=True,
    )
    indexer = glm5_2.Glm52Indexer(cfg, torch.bfloat16, torch.device("cpu"), layer_id=3)
    indexer.wq_b._set_dynamic_activation(False)
    generator = torch.Generator().manual_seed(128160)
    with torch.no_grad():
        indexer.wk.weight.copy_(torch.randn(128, 16, generator=generator))
        indexer.weights_proj.weight.copy_(torch.randn(32, 16, generator=generator))
        indexer.k_norm.weight.copy_(torch.randn(128, generator=generator))
        indexer.k_norm.bias.copy_(torch.randn(128, generator=generator))
    # CPU fakes exercise the dispatch contract, not NPU capability detection.
    indexer._direct_index_norm_layout = True
    with (
        patch.object(glm5_2.kernels, "prepare_quant_weight", side_effect=lambda weight: weight),
        patch.object(glm5_2.kernels, "supports_strided_indexer_layer_norm_out", return_value=True),
    ):
        indexer.process_weights_after_loading()
    return indexer


def _context(**options: bool) -> SimpleNamespace:
    return SimpleNamespace(
        execution_state=AclGraphExecutionState({}) if options.get("graph", True) else None,
        metadata=SimpleNamespace(
            is_prefill=options.get("prefill", False), is_chunked_prefill=options.get("chunked", False)
        ),
    )


def _normalize(
    projection: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
) -> None:
    key.copy_(F.layer_norm(projection[:, :128], (128,), gamma, beta, 1e-6))
    weights.copy_(projection[:, 128:])


@pytest.mark.parametrize("model_type", ("glm_moe_dsa", "glm_moe_dsa_mtp"))
@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
def test_target_and_draft_keep_live_outputs_independent_without_online_capability_probe(
    model_type: str, tokens: int
) -> None:
    indexer = _indexer(model_type)
    hidden = torch.ones(tokens, 16, dtype=torch.bfloat16)
    context = _context()
    index_context = SimpleNamespace(cp_context=None)
    with (
        forward_context(context),
        patch.object(glm5_2.kernels, "strided_indexer_layer_norm_out", side_effect=_normalize) as operation,
        patch.object(
            glm5_2.kernels, "supports_strided_indexer_layer_norm_out", side_effect=AssertionError("online probe")
        ),
    ):
        key, weights, normalized = indexer._project_normalized_index_inputs(hidden, hidden, index_context)
        first = key, weights
        snapshots = key.clone(), weights.clone()
        pointers = key.data_ptr(), weights.data_ptr()
        for factor in (2.0, -3.0):
            hidden.fill_(factor)
            key, weights, normalized = indexer._project_normalized_index_inputs(hidden, hidden, index_context)
            projection = F.linear(hidden, indexer._wk_weights_proj_weight)
            torch.testing.assert_close(key, indexer.k_norm(projection[:, :128]), rtol=0, atol=0)
            torch.testing.assert_close(weights, projection[:, 128:], rtol=0, atol=0)
            assert normalized
            assert key.data_ptr() != pointers[0] and weights.data_ptr() != pointers[1]
            for original, snapshot in zip(first, snapshots, strict=True):
                torch.testing.assert_close(original, snapshot, rtol=0, atol=0)
        assert operation.call_count == 3
    assert not context.execution_state.persistent_buffers


@pytest.mark.parametrize("model_type", ("glm_moe_dsa", "glm_moe_dsa_mtp"))
def test_consumed_indexer_outputs_are_not_retained_by_execution_state(model_type: str) -> None:
    indexer = _indexer(model_type)
    hidden = torch.ones(4, 16, dtype=torch.bfloat16)
    context = _context()
    metadata = torch.arange(4, dtype=torch.int64)
    context.execution_state.persistent_buffers[("STATIC_SLOTS",)] = metadata
    with (
        torch.no_grad(),
        forward_context(context),
        patch.object(glm5_2.kernels, "strided_indexer_layer_norm_out", new=_normalize),
    ):
        for _ in range(3):
            key, weights, normalized = indexer._project_normalized_index_inputs(
                hidden, hidden, SimpleNamespace(cp_context=None)
            )
            assert normalized
            references = weakref.ref(key), weakref.ref(weights)
            consumed_key, consumed_weights = key.clone(), weights.clone()
            del key, weights
            assert all(reference() is None for reference in references)
            assert consumed_key.shape == (4, 128)
            assert consumed_weights.shape == (4, 32)
    assert context.execution_state.persistent_buffers == {("STATIC_SLOTS",): metadata}


@pytest.mark.parametrize("model_type", ("glm_moe_dsa", "glm_moe_dsa_mtp"))
@pytest.mark.parametrize("tokens,steps", ((1, 3), (4, 1), (16, 1)))
def test_npu_consumed_indexer_outputs_replay_in_independent_graph_slots(
    model_type: str, tokens: int, steps: int
) -> None:
    if not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    torch.npu.set_device(0)
    indexer = _indexer(model_type).to("npu:0")
    hidden = [[torch.ones(tokens, 16, dtype=torch.bfloat16, device="npu:0") for _ in range(steps)] for _ in range(2)]
    contexts = [_context(), _context()]
    index_context = SimpleNamespace(cp_context=None)
    references = []

    def _consume(value: torch.Tensor) -> torch.Tensor:
        key, weights, normalized = indexer._project_normalized_index_inputs(value, value, index_context)
        assert normalized
        references.extend((weakref.ref(key), weakref.ref(weights)))
        return torch.cat((key, weights), dim=-1)

    graphs, results = [], []
    pool = torch.npu.graph_pool_handle()
    stream = torch.npu.Stream()
    with torch.no_grad():
        for slot, context in zip(hidden, contexts, strict=True):
            with forward_context(context):
                for _ in range(3):
                    for value in slot:
                        _consume(value)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                references.clear()
                with torch.npu.graph(graph, pool=pool, stream=stream):
                    result = torch.stack([_consume(value) for value in slot])
            assert all(reference() is None for reference in references)
            assert not context.execution_state.persistent_buffers
            graphs.append(graph)
            results.append(result)
        pointers = tuple(result.data_ptr() for result in results)
        # Prepared replay returns borrowed storage. Native MTP copies each
        # result into its next input before launching another shared-pool graph.
        consumed = [torch.empty_like(result) for result in results]
        for graph, result, output in zip(graphs, results, consumed, strict=True):
            graph.replay()
            output.copy_(result)
        torch.npu.synchronize()
        snapshots = tuple(output.cpu().clone() for output in consumed)
        for factor in (0.0, 1.0, -3.0):
            for slot_id, (slot, graph, result) in enumerate(zip(hidden, graphs, results, strict=True)):
                for step, value in enumerate(slot):
                    value.fill_(factor + step + slot_id)
                expected = []
                for value in slot:
                    projection = F.linear(value, indexer._wk_weights_proj_weight)
                    expected.append(torch.cat((indexer.k_norm(projection[:, :128]), projection[:, 128:]), dim=-1))
                expected_cpu = torch.stack(expected).cpu()
                graph.replay()
                consumed[slot_id].copy_(result)
                torch.npu.synchronize()
                torch.testing.assert_close(consumed[slot_id].cpu(), expected_cpu, rtol=0, atol=0)
                assert result.data_ptr() == pointers[slot_id]
                other = 1 - slot_id
                torch.testing.assert_close(consumed[other].cpu(), snapshots[other], rtol=0, atol=0)
                snapshots = tuple(output.cpu().clone() for output in consumed)
        for graph in graphs:
            graph.reset()


@pytest.mark.parametrize("path", ("eager", "prefill", "chunked", "cp", "distinct_rows", "bucket", "unprepared"))
def test_unsupported_paths_keep_native_normalization(path: str) -> None:
    indexer = _indexer()
    hidden = torch.ones(3 if path == "bucket" else 4, 16, dtype=torch.bfloat16)
    cache_hidden = hidden.clone() if path == "distinct_rows" else hidden
    context = _context(graph=path != "eager", prefill=path == "prefill", chunked=path == "chunked")
    index_context = SimpleNamespace(cp_context=object() if path == "cp" else None)
    if path == "unprepared":
        indexer._direct_index_norm_ready = False
    with (
        forward_context(context),
        patch.object(glm5_2.kernels, "strided_indexer_layer_norm_out", side_effect=AssertionError("unsupported route")),
    ):
        key, weights, normalized = indexer._project_normalized_index_inputs(hidden, cache_hidden, index_context)
    assert not normalized
    torch.testing.assert_close(key, F.linear(cache_hidden, indexer.wk.weight), rtol=0, atol=0)
    torch.testing.assert_close(weights, F.linear(hidden, indexer.weights_proj.weight), rtol=0, atol=0)


def test_reload_invalidates_affine_and_refreshes_bf16_checkpoint_values() -> None:
    indexer = _indexer()
    state = {name: value.clone() for name, value in indexer.state_dict().items()}
    state["k_norm.weight"].fill_(1.003)
    state["k_norm.bias"].fill_(-0.007)
    indexer.load_state_dict(state)
    assert not indexer._direct_index_norm_ready
    assert not indexer._wk_weights_proj_ready
    assert "_index_norm_gamma" not in state
    with (
        patch.object(glm5_2.kernels, "prepare_quant_weight", side_effect=lambda weight: weight),
        patch.object(glm5_2.kernels, "supports_strided_indexer_layer_norm_out", return_value=True),
    ):
        indexer.process_weights_after_loading()
    assert indexer._direct_index_norm_ready
    assert torch.equal(indexer._index_norm_gamma, indexer.k_norm.weight.float())
    assert torch.equal(indexer._index_norm_beta, indexer.k_norm.bias.float())


def test_failed_preparation_does_not_leave_ready_flags() -> None:
    indexer = _indexer()
    with (
        patch.object(indexer.wq_b, "process_weights_after_loading", side_effect=ValueError("reload failed")),
        pytest.raises(ValueError, match="reload failed"),
    ):
        indexer.process_weights_after_loading()
    assert not indexer._direct_index_norm_ready
    assert not indexer._wk_weights_proj_ready


@pytest.mark.parametrize("normalized", (False, True))
def test_cache_update_normalizes_exactly_once(normalized: bool) -> None:
    indexer = _indexer()
    projected = torch.ones(4, 128, dtype=torch.bfloat16)
    update = MagicMock()
    context = SimpleNamespace(
        cp_context=None,
        index_cache=torch.empty(1, dtype=torch.bfloat16),
        index_cache_scale=None,
        update_index_cache=update,
    )
    with (
        patch.object(indexer.k_norm, "forward", wraps=indexer.k_norm.forward) as layer_norm,
        patch.object(indexer, "_apply_interleaved_rope", side_effect=lambda value, angles: value),
    ):
        indexer._update_index_cache(projected, context, (projected, projected), projected, normalized)
    assert layer_norm.call_count == (0 if normalized else 1)
    update.assert_called_once()
    if normalized:
        assert update.call_args.args[0] is projected


def test_normalized_flag_requires_an_explicit_projected_key() -> None:
    indexer = _indexer()
    with pytest.raises(ValueError, match="supplied explicitly"):
        indexer._update_index_cache(torch.empty(1, 16), None, (None, None), projected_k_is_normalized=True)
