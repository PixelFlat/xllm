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

"""GLM direct Q-A dispatch, graph ownership, and complete preprocess replay."""

from __future__ import annotations

import struct
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.python.test_glm5_2_padded_uk import _attention, _context, _npu_inputs
from xllm.python.kernels_npu import mla
from xllm.python.model_executor.forward_context import forward_context
from xllm.python.models.deepseek_v32 import DeepseekV3MLAAttention
from xllm.python.models.glm5_2 import Glm52MLAAttention


@pytest.mark.parametrize("model_type", ("glm_moe_dsa", "glm_moe_dsa_mtp"))
@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
def test_target_and_draft_select_only_prepared_graph_route(model_type: str, tokens: int) -> None:
    attention = _attention(model_type)
    with forward_context(_context()):
        assert not attention._use_strided_q_a_quant_for(tokens)
        attention._decode_strided_q_a_ready = True
        with patch.object(mla, "supports_strided_q_a_quant_out", side_effect=AssertionError("online capability probe")):
            assert attention._use_strided_q_a_quant_for(tokens)
        assert not DeepseekV3MLAAttention._use_strided_q_a_quant_for(attention, tokens)


@pytest.mark.parametrize("options", ({"graph": False}, {"prefill": True}, {"chunked": True}, {"cp": True}))
def test_other_execution_paths_keep_native_quantization(options: dict[str, bool]) -> None:
    attention = _attention()
    attention._decode_strided_q_a_ready = True
    with forward_context(_context(**options)):
        assert not attention._use_strided_q_a_quant_for(4)


@pytest.mark.parametrize("tokens", (0, 3, 17, 32, 128))
def test_unsupported_token_buckets_keep_native_quantization(tokens: int) -> None:
    attention = _attention()
    attention._decode_strided_q_a_ready = True
    with forward_context(_context()):
        assert not attention._use_strided_q_a_quant_for(tokens)


@pytest.mark.parametrize(
    "overrides",
    (
        {"tokens": 3},
        {"kv_lora_rank": 256},
        {"q_lora_rank": 1536},
        {"qk_rope_head_dim": 32},
        {"epsilon": 1e-6},
        {"fuse_q_norm_quant": False},
    ),
)
def test_unsupported_explicit_geometry_is_rejected(overrides: dict[str, int | float | bool]) -> None:
    arguments = dict(
        tokens=4, kv_lora_rank=512, q_lora_rank=2048, qk_rope_head_dim=64, epsilon=1e-5, fuse_q_norm_quant=True
    )
    arguments.update(overrides)
    with pytest.raises(ValueError, match="geometry"):
        mla._validate_strided_q_a_quant_geometry(**arguments)


@pytest.mark.parametrize("epsilon", (1e-5, struct.unpack("f", struct.pack("f", 1e-5))[0]))
def test_cpp_float_epsilon_selects_the_same_fixed_kernel(epsilon: float) -> None:
    with patch.object(mla, "_STRIDED_Q_A_QUANT_OUT", object()):
        assert mla.supports_strided_q_a_quant_out(epsilon)
    mla._validate_strided_q_a_quant_geometry(4, 512, 2048, 64, epsilon, True)


@pytest.mark.parametrize("epsilon", (1e-6, 1.00000001e-5, float("nan"), float("inf"), 0.0))
def test_other_epsilon_values_do_not_select_the_fixed_kernel(epsilon: float) -> None:
    with patch.object(mla, "_STRIDED_Q_A_QUANT_OUT", object()):
        assert not mla.supports_strided_q_a_quant_out(epsilon)
    with pytest.raises(ValueError, match="geometry"):
        mla._validate_strided_q_a_quant_geometry(4, 512, 2048, 64, epsilon, True)


def test_failed_reload_clears_prepared_capability() -> None:
    attention = _attention()
    attention._decode_strided_q_a_ready = True
    with (
        patch.object(DeepseekV3MLAAttention, "process_weights_after_loading", side_effect=ValueError("reload failed")),
        pytest.raises(ValueError, match="reload failed"),
    ):
        attention.process_weights_after_loading()
    assert not attention._decode_strided_q_a_ready


@pytest.mark.parametrize(
    "invalid",
    (
        None,
        "capability",
        "dynamic",
        "consumer",
        "projection",
        "device",
        "dtype",
        "gamma",
        "stride",
        "epsilon",
        "cp",
        "dp",
        "split",
    ),
)
@pytest.mark.parametrize("epsilon", (1e-5, struct.unpack("f", struct.pack("f", 1e-5))[0]))
def test_preparation_selects_capability_once_for_exact_checkpoint_geometry(invalid: str | None, epsilon: float) -> None:
    weight = SimpleNamespace(
        device=SimpleNamespace(type="npu"), dtype=torch.bfloat16, shape=(2048,), is_contiguous=lambda: True
    )
    projection = SimpleNamespace(weight=SimpleNamespace(shape=(6144, 2624)))
    attention = SimpleNamespace(
        _decode_strided_q_a_ready=True,
        _use_fused_mla_decode=True,
        _dynamic_mla_ready=invalid != "dynamic",
        _can_fuse_q_norm_quant=lambda: invalid != "consumer",
        _a_projection=lambda: projection,
        q_a_layernorm=SimpleNamespace(weight=weight, eps=epsilon),
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        q_lora_rank=2048,
        cfg=SimpleNamespace(tp_size=16, dp_size=1, cp_size=1, layerwise_split_size=1),
        _attn_dp_layout=None,
    )
    if invalid == "projection":
        projection.weight.shape = (6144, 2048)
    elif invalid == "device":
        weight.device.type = "cpu"
    elif invalid == "dtype":
        weight.dtype = torch.float32
    elif invalid == "gamma":
        weight.shape = (1024,)
    elif invalid == "stride":
        weight.is_contiguous = lambda: False
    elif invalid == "epsilon":
        attention.q_a_layernorm.eps = 1e-6
    elif invalid == "cp":
        attention.cfg.cp_size = 2
    elif invalid == "dp":
        attention.cfg.dp_size = 2
    elif invalid == "split":
        attention.cfg.layerwise_split_size = 2
    operation = object() if invalid != "capability" else None
    with patch.object(mla, "_STRIDED_Q_A_QUANT_OUT", operation):
        Glm52MLAAttention._prepare_decode_q_a_quant(attention)
    assert attention._decode_strided_q_a_ready is (invalid is None)


def test_live_graph_outputs_are_independent_across_forwards_and_slots() -> None:
    projection = torch.ones(4, 2624, dtype=torch.bfloat16)
    gamma = torch.ones(2048, dtype=torch.bfloat16)
    reloaded_gamma = gamma.clone()
    contexts = [_context(), _context()]
    shared = {}
    for context in contexts:
        context.execution_state.shared_persistent_buffers = shared

    def _write(value: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, scales: torch.Tensor) -> None:
        assert torch.equal(value, projection)
        assert weight is gamma or weight is reloaded_gamma
        output.fill_(1)
        scales.fill_(2)

    with patch.object(mla, "_STRIDED_Q_A_QUANT_OUT", side_effect=_write) as operation:
        with forward_context(contexts[0]):
            first = mla._quantize_strided_q_a(projection, gamma)
            repeated = mla._quantize_strided_q_a(projection, gamma)
            layer = mla._quantize_strided_q_a(projection.clone(), gamma)
            reloaded = mla._quantize_strided_q_a(projection, reloaded_gamma)
        with forward_context(contexts[1]):
            slot = mla._quantize_strided_q_a(projection, gamma)
    assert operation.call_count == 5
    assert operation.call_args_list[3].args[1] is reloaded_gamma
    for distinct in (repeated, layer, reloaded, slot):
        assert all(left.data_ptr() != right.data_ptr() for left, right in zip(first, distinct, strict=True))
    assert all(not context.execution_state.persistent_buffers for context in contexts)
    assert not shared
    assert first[0].shape == (4, 2048) and first[0].dtype == torch.int8
    assert first[1].shape == (4,) and first[1].dtype == torch.float32


@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
def test_consumed_q_a_outputs_are_released_without_changing_owned_metadata(tokens: int) -> None:
    projection = torch.ones(tokens, 2624, dtype=torch.bfloat16)
    gamma = torch.ones(2048, dtype=torch.bfloat16)
    context = _context()
    metadata = torch.arange(tokens, dtype=torch.int64)
    context.execution_state.persistent_buffers[("STATIC_SLOTS",)] = metadata
    shared = {("DRAFT_WORKSPACE",): torch.empty(1)}
    context.execution_state.shared_persistent_buffers = shared

    def _write(value: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, scales: torch.Tensor) -> None:
        output.fill_(1)
        scales.fill_(2)

    with patch.object(mla, "_STRIDED_Q_A_QUANT_OUT", new=_write), forward_context(context):
        for _ in range(3):
            outputs = mla._quantize_strided_q_a(projection, gamma)
            references = tuple(weakref.ref(value) for value in outputs)
            consumed = outputs[0].float() * outputs[1].unsqueeze(-1)
            del outputs
            assert all(reference() is None for reference in references)
            assert torch.all(consumed == 2)
    assert context.execution_state.persistent_buffers == {("STATIC_SLOTS",): metadata}
    assert set(shared) == {("DRAFT_WORKSPACE",)}


@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("quantized_hidden", (False, True))
@pytest.mark.parametrize("epsilon", (1e-5, struct.unpack("f", struct.pack("f", 1e-5))[0]))
def test_npu_complete_preprocess_matches_native_for_target_and_mtp_shapes(
    tokens: int, quantized_hidden: bool, epsilon: float
) -> None:
    inputs = _npu_inputs(tokens)
    inputs["q_norm_epsilon"] = epsilon
    if quantized_hidden:
        inputs["hidden"], inputs["hidden_scale"] = mla.dynamic_quant(inputs["hidden"])
    cases = []
    for optimized in (False, True):
        arguments = dict(inputs)
        arguments.update(
            kv_cache=torch.zeros(2, 128, 512, dtype=torch.bfloat16, device=inputs["hidden"].device),
            rope_cache=torch.zeros(2, 128, 64, dtype=torch.bfloat16, device=inputs["hidden"].device),
            strided_q_a_quant=optimized,
        )
        context = _context(device=inputs["hidden"].device)
        for _ in range(3):
            with forward_context(context):
                mla.deepseek_mla_preprocess_decode_dynamic(**arguments)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph), forward_context(context):
            outputs = mla.deepseek_mla_preprocess_decode_dynamic(**arguments)
        cases.append((arguments, graph, outputs, tuple(value.data_ptr() for value in outputs)))

    generator = torch.Generator().manual_seed(10122048 + tokens)
    for update in range(3):
        if quantized_hidden:
            inputs["hidden"].copy_(
                torch.randint(-100, 101, inputs["hidden"].shape, generator=generator, dtype=torch.int8)
            )
            inputs["hidden_scale"].fill_(0.01 + update * 0.005)
        else:
            inputs["hidden"].copy_(torch.randn(inputs["hidden"].shape, generator=generator).bfloat16())
        inputs["q_norm_weight"].copy_(torch.randn(2048, generator=generator).bfloat16())
        angle = torch.randn(tokens, 1, 1, 64, generator=generator)
        inputs["rope_cos"].copy_(angle.cos())
        inputs["rope_sin"].copy_(angle.sin())
        inputs["slot_mapping"].copy_(torch.arange(tokens, dtype=torch.int32) + update * tokens)
        for _, graph, outputs, pointers in cases:
            graph.replay()
            assert tuple(value.data_ptr() for value in outputs) == pointers
        torch.npu.synchronize()
        for wanted, actual in zip(cases[0][2], cases[1][2], strict=True):
            assert torch.equal(actual.cpu(), wanted.cpu())
        for name in ("kv_cache", "rope_cache"):
            assert torch.equal(cases[0][0][name].cpu(), cases[1][0][name].cpu())
    for _, graph, _, _ in cases:
        graph.reset()
