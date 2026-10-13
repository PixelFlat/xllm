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

"""Direct KV selection and complete MLA replay with changing cache slots."""

from __future__ import annotations

import struct
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.python.test_glm5_2_padded_uk import _attention, _context, _npu_inputs
from tests.python.test_glm5_2_query_rope import _patterns
from xllm.python import kernels
from xllm.python.attention.backend import MlaPreprocessContext
from xllm.python.kernels_npu import mla
from xllm.python.model_executor.forward_context import forward_context
from xllm.python.models.deepseek_v32 import DeepseekV3MLAAttention
from xllm.python.models.glm5_2 import Glm52MLAAttention


@pytest.mark.parametrize("model_type", ("glm_moe_dsa", "glm_moe_dsa_mtp"))
@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
def test_target_and_draft_select_prepared_patterns_without_capability_probe(model_type: str, tokens: int) -> None:
    attention = _attention(model_type)
    patterns = _patterns(torch.device("cpu"))
    cache = torch.empty(1, 128, 512, dtype=torch.bfloat16)
    rope = torch.empty(1, 128, 64, dtype=torch.bfloat16)
    attention._decode_query_rope_patterns = patterns
    with forward_context(_context()), patch.object(kernels, "supports_strided_mla_kv_cache_layout", return_value=True):
        assert attention._decode_kv_rope_patterns_for(tokens, cache, rope) is None
        attention._decode_strided_kv_ready = True
        with patch.object(kernels, "supports_strided_mla_kv_cache_out", side_effect=AssertionError("online probe")):
            assert attention._decode_kv_rope_patterns_for(tokens, cache, rope) is patterns
        assert DeepseekV3MLAAttention._decode_kv_rope_patterns_for(attention, tokens, cache, rope) is None


@pytest.mark.parametrize("options", ({"graph": False}, {"prefill": True}, {"chunked": True}, {"cp": True}))
def test_non_decode_execution_keeps_native_writer(options: dict[str, bool]) -> None:
    attention = _attention()
    attention._decode_strided_kv_ready = True
    attention._decode_query_rope_patterns = _patterns(torch.device("cpu"))
    cache = torch.empty(1, 128, 512, dtype=torch.bfloat16)
    rope = torch.empty(1, 128, 64, dtype=torch.bfloat16)
    with (
        forward_context(_context(**options)),
        patch.object(kernels, "supports_strided_mla_kv_cache_layout", side_effect=AssertionError("unexpected probe")),
    ):
        assert attention._decode_kv_rope_patterns_for(4, cache, rope) is None


@pytest.mark.parametrize("tokens", (0, 3, 17, 128))
def test_unsupported_bucket_keeps_native_writer(tokens: int) -> None:
    attention = _attention()
    attention._decode_strided_kv_ready = True
    attention._decode_query_rope_patterns = _patterns(torch.device("cpu"))
    with forward_context(_context()):
        assert attention._decode_kv_rope_patterns_for(tokens, torch.empty(0), torch.empty(0)) is None


def test_unsupported_cache_layout_keeps_native_writer() -> None:
    attention = _attention()
    attention._decode_strided_kv_ready = True
    attention._decode_query_rope_patterns = _patterns(torch.device("cpu"))
    with forward_context(_context()), patch.object(kernels, "supports_strided_mla_kv_cache_layout", return_value=False):
        assert attention._decode_kv_rope_patterns_for(4, torch.empty(0), torch.empty(0)) is None


@pytest.mark.parametrize("model_type", ("glm_moe_dsa", "glm_moe_dsa_mtp"))
def test_model_preprocess_passes_prepared_patterns_and_shared_slots(model_type: str) -> None:
    attention = _attention(model_type)
    attention._decode_strided_kv_ready = True
    attention._decode_query_rope_patterns = _patterns(torch.device("cpu"))
    attention._combined_qkv = SimpleNamespace(
        _dynamic_activation=True, weight=torch.empty(0), weight_scale=torch.empty(0)
    )
    hidden = torch.ones(4, 16, dtype=torch.bfloat16)
    angle = torch.ones(4, 1, 1, 2, dtype=torch.bfloat16)
    slots = torch.arange(4, dtype=torch.int64)
    cache = MlaPreprocessContext(
        kv_cache=torch.empty(1, 128, 8),
        rope_cache=torch.empty(1, 128, 2),
        slot_mapping=slots.int(),
    )
    outputs = (torch.ones(4, 8), torch.ones(4), torch.ones(4, 2, 8), torch.ones(4, 2, 2))
    with (
        forward_context(_context()),
        patch.object(kernels, "supports_strided_mla_kv_cache_layout", return_value=True),
        patch(
            "xllm.python.models.deepseek_v32.kernels.deepseek_mla_preprocess_decode_dynamic",
            return_value=outputs,
        ) as preprocess,
    ):
        attention._preprocess_decode(hidden, angle, angle, cache, slot_mapping_int64=slots)
    assert preprocess.call_args.args[-1] is attention._decode_query_rope_patterns
    prepared = preprocess.call_args.args[22]
    assert prepared.dtype == torch.int64
    assert prepared.untyped_storage().data_ptr() == slots.untyped_storage().data_ptr()


@pytest.mark.parametrize("explicit_head", (False, True))
@pytest.mark.parametrize("format_name", ("ND", "NCHW", "FRACTAL_NZ"))
def test_layout_selection_accepts_only_flat_single_head_storage(explicit_head: bool, format_name: str) -> None:
    shape = (3, 128, 1, 512) if explicit_head else (3, 128, 512)
    device = SimpleNamespace(type="npu")
    cache = SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        device=device,
        dtype=torch.bfloat16,
        numel=lambda: 3 * 128 * 512,
        is_contiguous=lambda: True,
    )
    rope = SimpleNamespace(
        shape=(*shape[:-1], 64),
        ndim=len(shape),
        device=device,
        dtype=torch.bfloat16,
        is_contiguous=lambda: True,
    )
    with patch.object(mla.torch_npu, "get_npu_format", return_value=getattr(mla.torch_npu.Format, format_name)):
        assert mla.supports_strided_mla_kv_cache_layout(cache, rope) is (format_name != "FRACTAL_NZ")
        cache.numel = lambda: 2**31
        assert not mla.supports_strided_mla_kv_cache_layout(cache, rope)
        cache.numel = lambda: 3 * 128 * 512
        rope.is_contiguous = lambda: False
        assert not mla.supports_strided_mla_kv_cache_layout(cache, rope)


@pytest.mark.parametrize("epsilon", (1e-5, struct.unpack("f", struct.pack("f", 1e-5))[0]))
@pytest.mark.parametrize(
    "invalid", (None, "capability", "projection", "patterns", "device", "dtype", "width", "stride")
)
def test_preparation_checks_loaded_weight_and_capability_once(epsilon: float, invalid: str | None) -> None:
    weight = SimpleNamespace(
        device=SimpleNamespace(type="npu"), dtype=torch.bfloat16, shape=(512,), is_contiguous=lambda: True
    )
    attention = SimpleNamespace(
        _decode_strided_kv_ready=True,
        _decode_strided_q_a_ready=invalid != "projection",
        _decode_query_rope_patterns=None if invalid == "patterns" else object(),
        kv_a_layernorm=SimpleNamespace(weight=weight, eps=epsilon),
    )
    if invalid == "device":
        weight.device.type = "cpu"
    elif invalid == "dtype":
        weight.dtype = torch.float32
    elif invalid == "width":
        weight.shape = (256,)
    elif invalid == "stride":
        weight.is_contiguous = lambda: False
    with patch.object(mla, "_STRIDED_MLA_KV_CACHE_OUT", None if invalid == "capability" else object()):
        Glm52MLAAttention._prepare_decode_kv_cache(attention)
    assert attention._decode_strided_kv_ready is (invalid is None)


@pytest.mark.parametrize("epsilon", (1e-6, 1.00000001e-5, float("nan"), float("inf"), 0.0))
def test_other_epsilon_does_not_select_fixed_kernel(epsilon: float) -> None:
    with patch.object(mla, "_STRIDED_MLA_KV_CACHE_OUT", object()):
        assert not mla.supports_strided_mla_kv_cache_out(epsilon)
    with pytest.raises(ValueError, match="geometry"):
        mla._validate_strided_mla_kv_cache_geometry(4, 512, 2048, 64, epsilon)


def test_failed_reload_clears_direct_writer_and_patterns() -> None:
    attention = _attention()
    attention._decode_strided_kv_ready = True
    attention._decode_query_rope_patterns = _patterns(torch.device("cpu"))
    with (
        patch.object(DeepseekV3MLAAttention, "process_weights_after_loading", side_effect=ValueError("reload failed")),
        pytest.raises(ValueError, match="reload failed"),
    ):
        attention.process_weights_after_loading()
    assert not attention._decode_strided_kv_ready
    assert attention._decode_query_rope_patterns is None


@pytest.mark.parametrize("tokens", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("quantized_hidden", (False, True))
@pytest.mark.parametrize("prepared_slots", (False, True))
def test_npu_complete_preprocess_preserves_query_and_complete_cache(
    tokens: int, quantized_hidden: bool, prepared_slots: bool
) -> None:
    inputs = _npu_inputs(tokens)
    device = inputs["hidden"].device
    if quantized_hidden:
        inputs["hidden"], inputs["hidden_scale"] = mla.dynamic_quant(inputs["hidden"])
    inputs["strided_q_a_quant"] = True
    inputs["kv_norm_epsilon"] = struct.unpack("f", struct.pack("f", 1e-5))[0]
    if prepared_slots:
        inputs["slot_mapping_int64"] = inputs["slot_mapping"].long()
    cases = []
    for optimized in (False, True):
        arguments = dict(inputs)
        arguments.update(
            kv_cache=torch.full((3, 128, 512), 3.25, dtype=torch.bfloat16, device=device),
            rope_cache=torch.full((3, 128, 64), -2.5, dtype=torch.bfloat16, device=device),
            kv_rope_patterns=_patterns(device) if optimized else None,
        )
        context = _context(device=device)
        for _ in range(3):
            with forward_context(context):
                mla.deepseek_mla_preprocess_decode_dynamic(**arguments)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph), forward_context(context):
            outputs = mla.deepseek_mla_preprocess_decode_dynamic(**arguments)
        cases.append((arguments, graph, outputs, tuple(value.data_ptr() for value in outputs)))
    generator = torch.Generator().manual_seed(2624512 + tokens)
    for update, factor in enumerate((1.0, -0.5, 2.0)):
        if quantized_hidden:
            inputs["hidden"].copy_(
                torch.randint(-100, 101, inputs["hidden"].shape, generator=generator, dtype=torch.int8)
            )
            inputs["hidden_scale"].fill_(0.01 + update * 0.005)
        else:
            inputs["hidden"].copy_(torch.randn(inputs["hidden"].shape, generator=generator).bfloat16())
        inputs["kv_norm_weight"].copy_(torch.randn(512, generator=generator).bfloat16())
        angle = torch.randn(tokens, 1, 1, 64, generator=generator) * factor
        inputs["rope_cos"].copy_(angle.cos())
        inputs["rope_sin"].copy_(angle.sin())
        slots = torch.arange(tokens, dtype=torch.int32) + 127 + update * 32
        if update == 2:
            slots[-1] = -1
        inputs["slot_mapping"].copy_(slots)
        if prepared_slots:
            inputs["slot_mapping_int64"].copy_(slots)
        for arguments, graph, outputs, pointers in cases:
            arguments["kv_cache"].fill_(3.25)
            arguments["rope_cache"].fill_(-2.5)
            graph.replay()
            assert pointers == tuple(value.data_ptr() for value in outputs)
        torch.npu.synchronize()
        for wanted, actual in zip(cases[0][2], cases[1][2], strict=True):
            assert torch.equal(wanted.cpu(), actual.cpu())
        for name in ("kv_cache", "rope_cache"):
            assert torch.equal(cases[0][0][name].cpu(), cases[1][0][name].cpu())
    for _, graph, _, _ in cases:
        graph.reset()
