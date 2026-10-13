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

"""GLM query RoPE preserves interleaved semantics and graph replay lifetimes."""

from __future__ import annotations

import weakref
from typing import Any
from unittest.mock import patch

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from torch._subclasses.fake_tensor import FakeTensorMode

from tests.python.test_glm5_2_padded_uk import _attention, _context, _npu_inputs
from xllm.python.kernels_npu import mla
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.models.deepseek_v32 import DeepseekV3MLAAttention


def _patterns(device: torch.device) -> torch.Tensor:
    even = torch.arange(0, 64, 2, dtype=torch.int32)
    odd = even + 1
    return (torch.stack((torch.cat((even, odd)), torch.cat((odd, even)))) * 4).to(device)


@pytest.mark.parametrize("model_type", ["glm_moe_dsa", "glm_moe_dsa_mtp"])
def test_patterns_are_prepared_once_outside_graph_and_not_persistent(model_type: str) -> None:
    attention = _attention(model_type)
    assert attention._decode_query_rope_patterns is None
    with patch.object(attention, "_can_pad_decode_latent_weight", return_value=True):
        attention._prepare_decode_query_rope()
    patterns = attention._decode_query_rope_patterns
    torch.testing.assert_close(patterns, _patterns(torch.device("cpu")), rtol=0, atol=0)
    assert "_decode_query_rope_patterns" not in attention.state_dict()
    with forward_context(_context()):
        for tokens in (1, 4, 16):
            assert attention._decode_query_rope_patterns_for(tokens) is patterns
            assert DeepseekV3MLAAttention._decode_query_rope_patterns_for(attention, tokens) is None


@pytest.mark.parametrize("options", [{"graph": False}, {"prefill": True}, {"chunked": True}, {"cp": True}])
def test_non_decode_paths_do_not_use_query_kernel(options: dict[str, bool]) -> None:
    attention = _attention()
    attention._decode_query_rope_patterns = _patterns(torch.device("cpu"))
    with forward_context(_context(**options)):
        assert attention._decode_query_rope_patterns_for(4) is None


@pytest.mark.parametrize("tokens", [0, 2, 3, 8, 17, 128])
def test_unmeasured_decode_buckets_keep_native_rope(tokens: int) -> None:
    attention = _attention()
    attention._decode_query_rope_patterns = _patterns(torch.device("cpu"))
    with forward_context(_context()):
        assert attention._decode_query_rope_patterns_for(tokens) is None


def test_failed_reload_clears_old_gather_patterns() -> None:
    attention = _attention()
    attention._decode_query_rope_patterns = _patterns(torch.device("cpu"))
    with (
        patch.object(DeepseekV3MLAAttention, "process_weights_after_loading", side_effect=ValueError("reload failed")),
        pytest.raises(ValueError, match="reload failed"),
    ):
        attention.process_weights_after_loading()
    assert attention._decode_query_rope_patterns is None


def test_unsupported_geometry_clears_old_gather_patterns() -> None:
    attention = _attention()
    attention._decode_query_rope_patterns = _patterns(torch.device("cpu"))
    attention._prepare_decode_query_rope()
    assert attention._decode_query_rope_patterns is None


@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_fake_contract_preserves_supplied_output(tokens: int) -> None:
    with FakeTensorMode():
        q = torch.empty(tokens, 4, 256, dtype=torch.bfloat16, device="npu")
        coefficient = torch.empty(tokens, 1, 1, 64, dtype=q.dtype, device=q.device)
        patterns = torch.empty(2, 64, dtype=torch.int32, device=q.device)
        output = torch.empty(tokens, 4, 64, dtype=q.dtype, device=q.device)
        actual = torch.ops.xllm_ops.mla_query_rope_out(q, coefficient, coefficient, patterns, output)
        assert actual is output


@pytest.mark.parametrize("invalid", ["tokens", "dtype", "heads", "coefficients", "patterns", "output", "stride"])
def test_fake_contract_rejects_unsupported_geometry(invalid: str) -> None:
    with FakeTensorMode():
        q = torch.empty(4, 4, 256, dtype=torch.bfloat16, device="npu")
        coefficient = torch.empty(4, 1, 1, 64, dtype=q.dtype, device=q.device)
        patterns = torch.empty(2, 64, dtype=torch.int32, device=q.device)
        output = torch.empty(4, 4, 64, dtype=q.dtype, device=q.device)
        if invalid == "tokens":
            q = q[:2]
        elif invalid == "dtype":
            q = q.float()
        elif invalid == "heads":
            q = q[:, :2]
        elif invalid == "coefficients":
            coefficient = coefficient.view(4, 64)
        elif invalid == "patterns":
            patterns = patterns.to(torch.int64)
        elif invalid == "output":
            output = output[:, :2]
        elif invalid == "stride":
            q = torch.empty(4, 4, 512, dtype=q.dtype, device=q.device)[..., ::2]
        with pytest.raises(ValueError):
            torch.ops.xllm_ops.mla_query_rope_out(q, coefficient, coefficient, patterns, output)


@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_npu_query_rope_matches_native_and_preserves_inputs(tokens: int) -> None:
    if not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(1012)
    initial = torch.randn(tokens, 4, 256, generator=generator).bfloat16()
    query = initial.to("npu")
    angle = torch.randn(tokens, 1, 1, 64, generator=generator)
    cos, sin = angle.cos().bfloat16().npu(), angle.sin().bfloat16().npu()
    patterns = _patterns(query.device)
    output = torch.empty(tokens, 4, 64, dtype=query.dtype, device=query.device)
    for _ in range(3):
        torch.ops.xllm_ops.mla_query_rope_out(query, cos, sin, patterns, output)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        actual = torch.ops.xllm_ops.mla_query_rope_out(query, cos, sin, patterns, output)
    pointers = (query.data_ptr(), cos.data_ptr(), sin.data_ptr(), patterns.data_ptr(), output.data_ptr())
    for scale in (1.0, -0.5, 2.0):
        query.copy_(initial * scale)
        cos.copy_((angle * scale).cos())
        sin.copy_((angle * scale).sin())
        before = tuple(value.cpu().clone() for value in (query, cos, sin, patterns))
        expected = torch_npu.npu_interleave_rope(query[..., 192:].view(tokens, 4, 1, 64), cos, sin).view(tokens, 4, 64)
        graph.replay()
        torch.npu.synchronize()
        assert actual.data_ptr() == output.data_ptr()
        assert pointers == (query.data_ptr(), cos.data_ptr(), sin.data_ptr(), patterns.data_ptr(), output.data_ptr())
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        for value, unchanged in zip((query, cos, sin, patterns), before, strict=True):
            torch.testing.assert_close(value.cpu(), unchanged, rtol=0, atol=0)
    graph.reset()


@pytest.mark.parametrize("tokens", [1, 4, 16])
@pytest.mark.parametrize("quantized_hidden", [False, True])
def test_npu_complete_preprocess_matches_native_with_changed_graph_inputs(tokens: int, quantized_hidden: bool) -> None:
    inputs = _npu_inputs(tokens)
    if quantized_hidden:
        inputs["hidden"], inputs["hidden_scale"] = mla.dynamic_quant(inputs["hidden"])
    graph = None
    outputs = None
    cases = []
    for optimized in (False, True):
        arguments = dict(inputs)
        arguments.update(
            kv_cache=torch.zeros(2, 128, 512, dtype=torch.bfloat16, device=inputs["hidden"].device),
            rope_cache=torch.zeros(2, 128, 64, dtype=torch.bfloat16, device=inputs["hidden"].device),
            query_rope_patterns=_patterns(inputs["hidden"].device) if optimized else None,
        )
        if optimized:
            weight = inputs["w_uk"]
            arguments["w_uk"] = torch.cat((weight, weight.new_zeros(4, 64, 512)), dim=1)
        context = _context(device=inputs["hidden"].device)
        for _ in range(3):
            with forward_context(context):
                mla.deepseek_mla_preprocess_decode_dynamic(**arguments)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph), forward_context(context):
            outputs = mla.deepseek_mla_preprocess_decode_dynamic(**arguments)
        cases.append((arguments, graph, outputs, tuple(value.data_ptr() for value in outputs)))
    generator = torch.Generator().manual_seed(1013)
    for update in range(3):
        if quantized_hidden:
            inputs["hidden"].copy_(
                torch.randint(-100, 101, inputs["hidden"].shape, generator=generator, dtype=torch.int8)
            )
            inputs["hidden_scale"].fill_(0.01 + update * 0.005)
        else:
            inputs["hidden"].copy_(torch.randn(inputs["hidden"].shape, generator=generator).bfloat16())
        angle = torch.randn(tokens, 1, 1, 64, generator=generator)
        inputs["rope_cos"].copy_(angle.cos())
        inputs["rope_sin"].copy_(angle.sin())
        inputs["slot_mapping"].copy_(torch.arange(tokens, dtype=torch.int32) + update * tokens)
        for _, graph, outputs, pointers in cases:
            graph.replay()
            assert pointers == tuple(value.data_ptr() for value in outputs)
        torch.npu.synchronize()
        for left, right in zip(cases[0][2], cases[1][2], strict=True):
            torch.testing.assert_close(left.cpu(), right.cpu(), rtol=0, atol=0)
        for cache in ("kv_cache", "rope_cache"):
            torch.testing.assert_close(cases[0][0][cache].cpu(), cases[1][0][cache].cpu(), rtol=0, atol=0)
    for _, graph, _, _ in cases:
        graph.reset()


@pytest.mark.parametrize("tokens,steps", ((1, 3), (4, 1), (16, 1)))
def test_npu_preprocess_replays_after_consumed_intermediates_are_released(tokens: int, steps: int) -> None:
    inputs = _npu_inputs(tokens)
    inputs["w_uk"] = torch.cat((inputs["w_uk"], inputs["w_uk"].new_zeros(4, 64, 512)), dim=1)
    inputs["query_rope_patterns"] = _patterns(inputs["hidden"].device)
    inputs["kv_rope_patterns"] = inputs["query_rope_patterns"]
    inputs["strided_q_a_quant"] = True
    arguments = []
    contexts = []
    for _ in range(2):
        slot = []
        for step in range(steps):
            current = dict(inputs)
            for name in ("hidden", "rope_cos", "rope_sin", "slot_mapping"):
                current[name] = current[name].clone()
            current["slot_mapping"].add_(step * tokens)
            current["slot_mapping_int64"] = current["slot_mapping"].long()
            current["kv_cache"] = inputs["hidden"].new_zeros(2, 128, 512)
            current["rope_cache"] = inputs["hidden"].new_zeros(2, 128, 64)
            slot.append(current)
        arguments.append(slot)
        context = _context(device=inputs["hidden"].device)
        context.execution_state.persistent_buffers[("STATIC_SLOTS",)] = slot[0]["slot_mapping_int64"]
        contexts.append(context)

    references = []

    def _consume(current: dict[str, Any], context: ForwardContext) -> torch.Tensor:
        with forward_context(context):
            outputs = mla.deepseek_mla_preprocess_decode_dynamic(**current)
        references.extend(weakref.ref(value) for value in outputs)
        q_a = outputs[0].float() * outputs[1].unsqueeze(-1)
        return torch.cat((q_a, outputs[2].flatten(1).float(), outputs[3].flatten(1).float()), dim=-1)

    graphs, results = [], []
    pool = torch.npu.graph_pool_handle()
    stream = torch.npu.Stream()
    for slot, context in zip(arguments, contexts, strict=True):
        for _ in range(3):
            for current in slot:
                _consume(current, context)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        references.clear()
        with torch.npu.graph(graph, pool=pool, stream=stream):
            result = torch.stack([_consume(current, context) for current in slot])
        assert all(reference() is None for reference in references)
        assert set(context.execution_state.persistent_buffers) == {("STATIC_SLOTS",)}
        graphs.append(graph)
        results.append(result)
    pointers = tuple(result.data_ptr() for result in results)
    # Native prepared MTP consumes borrowed graph outputs before another
    # shared-pool replay; only its input/output Slot storage must remain intact.
    consumed = [torch.empty_like(result) for result in results]
    for graph, result, output in zip(graphs, results, consumed, strict=True):
        graph.replay()
        output.copy_(result)
    torch.npu.synchronize()
    snapshots = tuple(output.cpu().clone() for output in consumed)
    for update in range(3):
        for slot_id, (slot, context, graph, result) in enumerate(
            zip(arguments, contexts, graphs, results, strict=True)
        ):
            for step, current in enumerate(slot):
                current["hidden"].fill_(0.25 * (1 + update + step + slot_id))
                current["rope_cos"].fill_(0.5 + 0.25 * update)
                current["rope_sin"].fill_(0.125 * slot_id)
            expected_cpu = torch.stack([_consume(current, context) for current in slot]).cpu()
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
