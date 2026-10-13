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

"""Decode-only padding contracts for GLM target and draft latent projection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("torch_npu")

from xllm.python.attention.backend import MlaPreprocessContext
from xllm.python.kernels_npu import mla
from xllm.python.kernels_npu.linear import prepare_quant_weight
from xllm.python.model_executor.forward_context import AclGraphExecutionState, ForwardContext, forward_context
from xllm.python.models.deepseek_v32 import DeepseekV3MLAAttention
from xllm.python.models.glm5_2 import Glm52Config, Glm52MLAAttention


def _attention(model_type: str = "glm_moe_dsa") -> Glm52MLAAttention:
    cfg = Glm52Config.from_dict(
        {
            "model_type": model_type,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "q_lora_rank": 8,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 2,
            "v_head_dim": 4,
            "index_n_heads": 2,
            "index_head_dim": 8,
            "index_topk": 4,
            "indexer_types": ["shared"],
            "tp_size": 1,
            "dp_size": 1,
            "cp_size": 1,
        }
    )
    return Glm52MLAAttention(cfg, 0, torch.bfloat16, torch.device("cpu"))


def _context(
    *,
    graph: bool = True,
    prefill: bool = False,
    chunked: bool = False,
    cp: bool = False,
    device: torch.device | None = None,
) -> ForwardContext:
    return ForwardContext(
        attention_backend=None,
        device=device if device is not None else torch.device("cpu"),
        metadata=SimpleNamespace(is_prefill=prefill, is_chunked_prefill=chunked),
        layer_caches=[],
        execution_state=AclGraphExecutionState({}) if graph else None,
        cp_context=SimpleNamespace() if cp else None,
    )


@pytest.mark.parametrize("model_type", ["glm_moe_dsa", "glm_moe_dsa_mtp"])
def test_padding_is_nonpersistent_and_keeps_original_weight(model_type: str) -> None:
    attention = _attention(model_type)
    assert attention._decode_w_uk is None
    attention.W_UK.fill_(2)
    original = attention.W_UK.clone()
    with patch.object(attention, "_can_pad_decode_latent_weight", return_value=True):
        attention._prepare_decode_latent_weight()

    torch.testing.assert_close(attention.W_UK, original, rtol=0, atol=0)
    torch.testing.assert_close(attention._decode_w_uk[:, :4], original, rtol=0, atol=0)
    assert torch.count_nonzero(attention._decode_w_uk[:, 4:]) == 0
    assert attention._decode_w_uk.is_contiguous()
    assert "_decode_w_uk" not in attention.state_dict()


@pytest.mark.parametrize("model_type", ["glm_moe_dsa", "glm_moe_dsa_mtp"])
@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_graph_target_and_draft_select_padded_weight(model_type: str, tokens: int) -> None:
    attention = _attention(model_type)
    attention._decode_w_uk = torch.zeros(2, 6, 8, dtype=torch.bfloat16)
    with forward_context(_context()):
        assert attention._decode_latent_weight(tokens) is attention._decode_w_uk
        assert DeepseekV3MLAAttention._decode_latent_weight(attention, tokens) is attention.W_UK


@pytest.mark.parametrize("options", [{"graph": False}, {"prefill": True}, {"chunked": True}, {"cp": True}])
def test_non_decode_paths_keep_original_weight(options: dict[str, bool]) -> None:
    attention = _attention()
    attention._decode_w_uk = torch.zeros(2, 6, 8, dtype=torch.bfloat16)
    with forward_context(_context(**options)):
        assert attention._decode_latent_weight(4) is attention.W_UK


@pytest.mark.parametrize("tokens", [0, 17, 128])
def test_large_or_empty_graph_keeps_original_weight(tokens: int) -> None:
    attention = _attention()
    attention._decode_w_uk = torch.zeros(2, 6, 8, dtype=torch.bfloat16)
    with forward_context(_context()):
        assert attention._decode_latent_weight(tokens) is attention.W_UK


@pytest.mark.parametrize("model_type", ["glm_moe_dsa", "glm_moe_dsa_mtp"])
def test_dynamic_preprocess_passes_selected_decode_weight(model_type: str) -> None:
    attention = _attention(model_type)
    attention._decode_w_uk = torch.zeros(2, 6, 8, dtype=torch.bfloat16)
    attention._combined_qkv = SimpleNamespace(
        _dynamic_activation=True, weight=torch.empty(0), weight_scale=torch.empty(0)
    )
    hidden = torch.ones(4, 16, dtype=torch.bfloat16)
    rope = torch.ones(4, 1, 1, 2, dtype=torch.bfloat16)
    cache = MlaPreprocessContext(
        kv_cache=torch.empty(1, 128, 8),
        rope_cache=torch.empty(1, 128, 2),
        slot_mapping=torch.arange(4, dtype=torch.int32),
    )
    outputs = (torch.ones(4, 8), torch.ones(4), torch.ones(4, 2, 8), torch.ones(4, 2, 2))
    with (
        forward_context(_context()),
        patch(
            "xllm.python.models.deepseek_v32.kernels.deepseek_mla_preprocess_decode_dynamic",
            return_value=outputs,
            create=True,
        ) as preprocess,
    ):
        attention._preprocess_decode(hidden, rope, rope, cache)
    assert preprocess.call_args.args[6] is attention._decode_w_uk


def test_reload_rebuilds_padding_from_new_weight() -> None:
    attention = _attention()
    attention._decode_w_uk = torch.full((2, 6, 8), -1, dtype=torch.bfloat16)

    def _load_weight() -> None:
        assert attention._decode_w_uk is None
        attention.W_UK.fill_(3)

    with (
        patch.object(DeepseekV3MLAAttention, "process_weights_after_loading", side_effect=_load_weight),
        patch.object(attention, "_can_pad_decode_latent_weight", return_value=True),
    ):
        attention.process_weights_after_loading()
    assert torch.all(attention._decode_w_uk[:, :4] == 3)
    assert torch.count_nonzero(attention._decode_w_uk[:, 4:]) == 0


def test_failed_reload_clears_old_padding_and_propagates_error() -> None:
    attention = _attention()
    attention._decode_w_uk = torch.zeros(2, 6, 8, dtype=torch.bfloat16)
    with (
        patch.object(DeepseekV3MLAAttention, "process_weights_after_loading", side_effect=ValueError("reload failed")),
        pytest.raises(ValueError, match="reload failed"),
    ):
        attention.process_weights_after_loading()
    assert attention._decode_w_uk is None


def test_unsupported_reload_does_not_retain_padding() -> None:
    attention = _attention()
    attention._decode_w_uk = torch.zeros(2, 6, 8, dtype=torch.bfloat16)
    attention._prepare_decode_latent_weight()
    assert attention._decode_w_uk is None


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"device": "cpu"},
        {"dtype": torch.float32},
        {"shape": (8, 192, 512)},
        {"q_lora_rank": 1536},
        {"qk_rope_head_dim": 32},
        {"tp_size": 8},
        {"dp_size": 2},
        {"cp_size": 2},
        {"layerwise_split_size": 2},
        {"_dynamic_mla_ready": False},
        {"_use_fused_mla_decode": False},
        {"_attn_dp_layout": object()},
    ],
)
def test_padding_capability_is_narrow(overrides: dict[str, object]) -> None:
    values = {
        "device": "npu",
        "dtype": torch.bfloat16,
        "shape": (4, 192, 512),
        "q_lora_rank": 2048,
        "qk_rope_head_dim": 64,
        "tp_size": 16,
        "dp_size": 1,
        "cp_size": 1,
        "layerwise_split_size": 1,
        "_use_fused_mla_decode": True,
        "_dynamic_mla_ready": True,
        "_attn_dp_layout": None,
    }
    values.update(overrides)
    attention = SimpleNamespace(
        **{
            name: values[name]
            for name in (
                "q_lora_rank",
                "qk_rope_head_dim",
                "_use_fused_mla_decode",
                "_dynamic_mla_ready",
                "_attn_dp_layout",
            )
        },
        cfg=SimpleNamespace(
            **{name: values[name] for name in ("tp_size", "dp_size", "cp_size", "layerwise_split_size")}
        ),
        W_UK=SimpleNamespace(
            device=SimpleNamespace(type=values["device"]), dtype=values["dtype"], shape=values["shape"]
        ),
    )
    assert Glm52MLAAttention._can_pad_decode_latent_weight(attention) is (not overrides)


@pytest.mark.parametrize("tokens", [1, 4, 16])
def test_padded_projection_consumes_full_query_without_nope_copy(tokens: int) -> None:
    generator = torch.Generator().manual_seed(1)
    query = torch.randint(-3, 4, (tokens, 2, 6), generator=generator).to(torch.bfloat16)
    weight = torch.randint(-3, 4, (2, 4, 8), generator=generator).to(torch.bfloat16)
    padded = torch.cat((weight, torch.zeros(2, 2, 8, dtype=torch.bfloat16)), dim=1)

    def _matmul(value: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return torch.einsum("thd,hdo->tho", value, rhs)

    with patch.object(mla, "atb_matmul_ein_sum", side_effect=_matmul) as projection:
        expected, expected_rope = mla._project_decode_latent_query(query, weight, 4, 2)
        actual, actual_rope = mla._project_decode_latent_query(query, padded, 4, 2)
    assert projection.call_args.args[0] is query
    assert projection.call_args.args[0].is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_rope, expected_rope, rtol=0, atol=0)
    assert actual_rope.untyped_storage().data_ptr() == query.untyped_storage().data_ptr()


@pytest.mark.parametrize("shape", [(2, 5, 8), (3, 4, 8), (2, 4, 7), (4, 8)])
def test_invalid_latent_weight_is_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="decode latent weight"):
        mla._validate_decode_latent_weight(torch.empty(shape), 2, 4, 2, 8)


@pytest.mark.parametrize("width", [4, 6])
def test_fake_preprocess_keeps_output_contract(width: int) -> None:
    hidden = torch.empty(4, 16)
    unused = torch.empty(0)
    outputs = mla._deepseek_mla_preprocess_decode_dynamic_fake(
        hidden,
        unused,
        unused,
        unused,
        unused,
        unused,
        torch.empty(2, width, 8),
        unused,
        unused,
        unused,
        unused,
        unused,
        unused,
        8,
        8,
        2,
        4,
        2,
        1e-5,
        1e-5,
        True,
    )
    assert [tuple(output.shape) for output in outputs] == [(4, 8), (4,), (4, 2, 8), (4, 2, 2)]
    assert [output.dtype for output in outputs] == [torch.int8, torch.float32, torch.bfloat16, torch.bfloat16]


def _npu_inputs(tokens: int) -> dict[str, Any]:
    if not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    generator = torch.Generator().manual_seed(9)
    device = torch.device("npu")

    def _quant_weight(rows: int, columns: int) -> torch.Tensor:
        value = torch.randint(-4, 5, (columns, rows), dtype=torch.int8, generator=generator)
        return prepare_quant_weight(value.to(device))

    # Real GLM geometry, with deterministic synthetic activations and weights.
    return {
        "hidden": torch.randn(tokens, 6144, generator=generator).to(device=device, dtype=torch.bfloat16),
        "qkv_weight": _quant_weight(6144, 2624),
        "qkv_weight_scale": torch.full((2624,), 0.001, device=device),
        "q_norm_weight": torch.ones(2048, dtype=torch.bfloat16, device=device),
        "q_b_weight": _quant_weight(2048, 1024),
        "q_b_weight_scale": torch.full((1024,), 0.001, device=device),
        "w_uk": torch.randn(4, 192, 512, generator=generator).to(device=device, dtype=torch.bfloat16),
        "kv_norm_weight": torch.ones(512, dtype=torch.bfloat16, device=device),
        "rope_cos": torch.ones(tokens, 1, 1, 64, dtype=torch.bfloat16, device=device),
        "rope_sin": torch.zeros(tokens, 1, 1, 64, dtype=torch.bfloat16, device=device),
        "slot_mapping": torch.arange(tokens, dtype=torch.int32, device=device),
        "kv_lora_rank": 512,
        "q_lora_rank": 2048,
        "num_heads": 4,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "q_norm_epsilon": 1e-5,
        "kv_norm_epsilon": 1e-5,
        "fuse_q_norm_quant": True,
    }


@pytest.mark.parametrize("tokens", [1, 4, 16])
@pytest.mark.parametrize("quantized_hidden", [False, True])
def test_npu_complete_preprocess_graph_replay_preserves_outputs_and_cache(tokens: int, quantized_hidden: bool) -> None:
    inputs = _npu_inputs(tokens)
    if quantized_hidden:
        inputs["hidden"], inputs["hidden_scale"] = mla.dynamic_quant(inputs["hidden"])
    padded = torch.cat((inputs["w_uk"], inputs["w_uk"].new_zeros(4, 64, 512)), dim=1)
    cases: list[tuple[dict[str, Any], ForwardContext, object, tuple[torch.Tensor, ...], tuple[int, ...]]] = []
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for weight in (inputs["w_uk"], padded):
            arguments = dict(inputs)
            arguments.update(
                w_uk=weight,
                kv_cache=torch.zeros(2, 128, 512, dtype=torch.bfloat16, device=weight.device),
                rope_cache=torch.zeros(2, 128, 64, dtype=torch.bfloat16, device=weight.device),
            )
            context = _context(device=weight.device)
            for _ in range(2):
                with forward_context(context):
                    mla.deepseek_mla_preprocess_decode_dynamic(**arguments)
            stream.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, stream=stream), forward_context(context):
                outputs = mla.deepseek_mla_preprocess_decode_dynamic(**arguments)
            cases.append((arguments, context, graph, outputs, tuple(value.data_ptr() for value in outputs)))
    torch.npu.current_stream().wait_stream(stream)

    generator = torch.Generator().manual_seed(10)
    for update in range(3):
        if quantized_hidden:
            inputs["hidden"].copy_(torch.randint(-100, 101, (tokens, 6144), generator=generator, dtype=torch.int8))
            inputs["hidden_scale"].fill_(0.01 + update * 0.005)
        else:
            inputs["hidden"].copy_(torch.randn(tokens, 6144, generator=generator).to(dtype=torch.bfloat16))
        angles = torch.randn(tokens, 1, 1, 64, generator=generator)
        inputs["rope_cos"].copy_(angles.cos().to(dtype=torch.bfloat16))
        inputs["rope_sin"].copy_(angles.sin().to(dtype=torch.bfloat16))
        inputs["slot_mapping"].copy_(torch.arange(tokens, dtype=torch.int32) + update * tokens)
        for _, _, graph, outputs, pointers in cases:
            graph.replay()
            assert tuple(value.data_ptr() for value in outputs) == pointers
        torch.npu.synchronize()
        for expected, actual in zip(cases[0][3], cases[1][3], strict=True):
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        for cache_name in ("kv_cache", "rope_cache"):
            torch.testing.assert_close(cases[1][0][cache_name].cpu(), cases[0][0][cache_name].cpu(), rtol=0, atol=0)
