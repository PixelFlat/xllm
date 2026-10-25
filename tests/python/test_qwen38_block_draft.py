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

"""Qwen3.8 target capture and block-verification state contracts."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from tests.python.qwen3_5_test_utils import make_linear_config
from xllm.python import kernels
from xllm.python.attention.backend import LayerCache, LinearVerifyGraphInputs
from xllm.python.layers.flash_comm1 import FlashComm1Context
from xllm.python.layers.npu.qwen3_5.gated_delta_net import NpuQwen3_5GatedDeltaNet
from xllm.python.layers.npu.qwen3_5.gdn_metadata import GdnMetadata, GdnPrefillMetadata, GdnVerifyMetadata
from xllm.python.layers.npu.qwen3_5.gdn_metadata_builder import Qwen3_5GdnMetadataBuilder
from xllm.python.layers.rotary_embedding import RotaryEmbedding
from xllm.python.model_executor.dspark_graph_input import DSparkAdaptiveGraphInputs
from xllm.python.model_executor.forward_context import ForwardContext, forward_context
from xllm.python.model_executor.runners.prepared_acl_graph import PreparedAclGraphRunner
from xllm.python.models import qwen3_5
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.qwen3 import Qwen3Config
from xllm.python.models.qwen3_5 import Qwen3_5Model
from xllm.python.models.qwen3_dflash2 import DFlash2Qwen3Config
from xllm.python.models.qwen3_dspark import Qwen3DSparkConfig


class _ResidualLayer(nn.Module):
    def __init__(self, delta: float) -> None:
        super().__init__()
        self.delta = delta

    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor | None, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions
        residual = hidden if residual is None else hidden + residual
        return torch.full_like(hidden, self.delta), residual


class _FinalNorm(nn.Module):
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (hidden + residual) * 0.5, residual


@pytest.mark.parametrize("capture_layers", [(), (2, 0)])
def test_target_captures_unnormalized_post_layer_streams(capture_layers: tuple[int, ...]) -> None:
    model = Qwen3_5Model.__new__(Qwen3_5Model)
    nn.Module.__init__(model)
    model._inputs_embeds = None
    model.embed_tokens = nn.Identity()
    model.layers = nn.ModuleList(_ResidualLayer(delta) for delta in (1.0, 2.0, 3.0))
    model.norm = _FinalNorm()
    model.aux_hidden_capture = AuxHiddenCapture(capture_layers)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    actual = model(hidden, torch.arange(2))

    if not capture_layers:
        torch.testing.assert_close(actual, (hidden + 6.0) * 0.5)
    else:
        output, captured = actual
        torch.testing.assert_close(output, (hidden + 6.0) * 0.5)
        torch.testing.assert_close(captured, torch.cat((hidden + 6.0, hidden + 1.0), dim=-1))


@pytest.mark.parametrize("capture_layers", [(), (2, 0)])
def test_target_capture_gathers_flashcomm1_rows(
    monkeypatch: pytest.MonkeyPatch, capture_layers: tuple[int, ...]
) -> None:
    model = Qwen3_5Model.__new__(Qwen3_5Model)
    nn.Module.__init__(model)
    model._inputs_embeds = None
    model.embed_tokens = nn.Identity()
    model.layers = nn.ModuleList(_ResidualLayer(delta) for delta in (1.0, 2.0, 3.0))
    model.norm = _FinalNorm()
    model.aux_hidden_capture = AuxHiddenCapture(capture_layers)
    context = FlashComm1Context(num_tokens=2, local_num_tokens=1, tp_size=2, tp_rank=0)
    monkeypatch.setattr(qwen3_5, "get_flash_comm1_context", lambda: context)
    monkeypatch.setattr(
        "xllm.python.distributed.tp_all_gather",
        lambda hidden, dim, world_size: torch.cat([hidden] * world_size, dim=dim),
    )
    hidden = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    actual = model(hidden, torch.arange(2))
    if capture_layers:
        output, captured = actual
        torch.testing.assert_close(output, (hidden + 6.0) * 0.5)
        torch.testing.assert_close(captured, torch.cat((hidden + 6.0, hidden + 1.0), dim=-1))
    else:
        torch.testing.assert_close(actual, (hidden + 6.0) * 0.5)


def _draft_config() -> dict:
    return {
        "hidden_size": 5120,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 17408,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000},
    }


def test_incoai_qwen38_dflash2_geometry() -> None:
    config = _draft_config()
    config.update(
        sliding_window=2048,
        dflash_config={
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "selector_rank": 256,
            "selector_top_k": 16,
        },
    )
    cfg = DFlash2Qwen3Config.from_dict(config)
    cfg.validate()
    assert cfg.rope_theta == 10000000
    assert cfg.block_size == 8
    assert cfg.head_split() == (32, 8)


def test_reflected_legacy_rope_scaling_is_not_an_hf_dictionary() -> None:
    assert Qwen3Config.from_dict({"rope_scaling": -1}).rope_scaling == {}


def test_radixark_qwen38_dspark_preserves_yarn_and_null_window() -> None:
    config = _draft_config()
    config.update(
        markov_rank=256,
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        sliding_window=None,
        rope_scaling={"rope_type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192},
    )
    cfg = Qwen3DSparkConfig.from_dict(config)
    cfg.validate()
    assert cfg.sliding_window == 0
    assert cfg.rope_scaling["factor"] == 32.0
    assert cfg.enable_confidence_head


def test_reflected_dspark_rope_matches_hf_rope() -> None:
    config = _draft_config()
    config.update(
        markov_rank=256,
        rope_scaling_rope_type="yarn",
        rope_scaling_factor=32.0,
        rope_scaling_original_max_position_embeddings=8192,
        rope_scaling_beta_fast=32,
        rope_scaling_beta_slow=1,
        rope_scaling_attn_factor=0.0,
    )
    cfg = Qwen3DSparkConfig.from_dict(config)
    cache = RotaryEmbedding(128, 32, cfg.rope_theta, rope_scaling=cfg.rope_scaling).cos_sin_cache
    expected = RotaryEmbedding(
        128,
        32,
        cfg.rope_theta,
        rope_scaling={"rope_type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192},
    ).cos_sin_cache
    torch.testing.assert_close(cache, expected)
    torch.testing.assert_close(cache[0, :64], torch.full((64,), 1.0 + 0.1 * math.log(32.0)))
    assert not torch.equal(cache[31], RotaryEmbedding(128, 32, cfg.rope_theta).cos_sin_cache[31])


def _verify_builder() -> Qwen3_5GdnMetadataBuilder:
    cfg = make_linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    builder = Qwen3_5GdnMetadataBuilder(cfg)
    builder.bind_layer_caches(
        [
            LayerCache(
                key=None,
                value=None,
                conv=torch.zeros(3, 10, 384, dtype=torch.bfloat16),
                ssm=torch.zeros(24, 1, 128, 128),
            )
        ]
    )
    return builder


@pytest.mark.parametrize("end_only", [False, True])
def test_prefill_accepts_native_slot_cumulative_ends(end_only: bool) -> None:
    ends = [2, 5] if end_only else [0, 2, 5]
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(5, dtype=torch.int32),
        linear_state_indices=torch.tensor([1, 2], dtype=torch.int32),
        linear_state_read_indices=None,
        linear_state_write_indices=None,
        is_dummy=False,
        is_spec_verify=False,
        is_prefill=True,
        is_chunked_prefill=False,
        q_seq_lens_host=torch.tensor([2, 3], dtype=torch.int32),
        q_cu_seq_lens=torch.tensor(ends, dtype=torch.int32),
        q_cu_seq_lens_host_values=ends,
        has_initial_state=torch.tensor([False, True]),
    )
    actual = _verify_builder().build(metadata)
    assert isinstance(actual, GdnPrefillMetadata)
    torch.testing.assert_close(actual.cu_seqlens, torch.tensor([0, 2, 5], dtype=torch.int32))
    if end_only:
        metadata.q_cu_seq_lens[0] = 1
        with pytest.raises(ValueError, match="do not match host"):
            _verify_builder().build(metadata)


@pytest.mark.parametrize("expanded", [False, True])
def test_verify_metadata_keeps_previous_accepted_checkpoint(expanded: bool) -> None:
    states = torch.tensor([1, 2], dtype=torch.int32)
    counts = torch.tensor([1, 5], dtype=torch.int32)
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(16, dtype=torch.int32),
        q_seq_lens_host=torch.tensor([8, 8], dtype=torch.int32),
        linear_state_indices=states.repeat_interleave(8) if expanded else states,
        num_accepted_tokens=counts,
        num_accepted_tokens_host_values=(1, 5),
        is_dummy=False,
        is_spec_verify=True,
    )
    actual = _verify_builder().build(metadata)
    assert isinstance(actual, GdnVerifyMetadata)
    assert actual.query_width == 8
    torch.testing.assert_close(actual.state_indices, states)
    torch.testing.assert_close(actual.num_accepted_tokens, counts)
    if not expanded:
        assert actual.num_accepted_tokens is counts
        assert actual.state_indices is states


def test_verify_rejects_query_width_exceeding_checkpoint_stride() -> None:
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(18, dtype=torch.int32),
        q_seq_lens_host=torch.tensor([9, 9], dtype=torch.int32),
        linear_state_indices=torch.tensor([1, 2], dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        num_accepted_tokens_host_values=(1, 1),
        is_dummy=False,
        is_spec_verify=True,
    )
    with pytest.raises(ValueError, match="checkpoint stride"):
        _verify_builder().build(metadata)


def test_verify_graph_uses_owned_device_maps_without_host_lengths() -> None:
    graph = LinearVerifyGraphInputs(
        torch.tensor([2, 1, 0], dtype=torch.int32),
        torch.tensor([3, 5, 1], dtype=torch.int32),
        8,
        torch.tensor([0, 1, 8, 9, 10, 24, 24, 24], dtype=torch.int64),
        torch.full((24,), 8, dtype=torch.int64),
    )
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(8, dtype=torch.int32),
        is_dummy=False,
        linear_verify_graph=graph,
    )
    actual = _verify_builder().build(metadata)
    assert isinstance(actual, GdnVerifyMetadata)
    assert actual.state_indices is graph.state_indices
    assert actual.num_accepted_tokens is graph.num_accepted_tokens
    assert actual.packed_to_dense is graph.packed_to_dense
    assert actual.dense_to_packed is graph.dense_to_packed
    assert actual.query_width == 8


def test_verify_routes_to_checkpointed_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = make_linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, torch.device("cpu"))
    layer.out_proj = nn.Identity()
    builder = _verify_builder()
    states = torch.tensor([1, 2], dtype=torch.int32)
    metadata = builder.build(
        SimpleNamespace(
            slot_mapping=torch.arange(16, dtype=torch.int32),
            q_seq_lens_host=torch.tensor([8, 8], dtype=torch.int32),
            linear_state_indices=states,
            num_accepted_tokens=torch.tensor([2, 7], dtype=torch.int32),
            num_accepted_tokens_host_values=(2, 7),
            is_dummy=False,
            is_spec_verify=True,
        )
    )
    verify = MagicMock(return_value=torch.ones(2, 8, 1, 128, dtype=torch.bfloat16))
    monkeypatch.setattr(kernels, "mega_gdn_mtp_decode", verify, raising=False)
    context = ForwardContext(None, torch.device("cpu"), None, [])
    context.layer_shared_cache[GdnMetadata] = metadata
    with forward_context(context):
        output = layer(torch.zeros(16, 8, dtype=torch.bfloat16))
    assert output.shape == (16, 128)
    args = verify.call_args.args
    assert args[0].shape == (2, 8, 384)
    torch.testing.assert_close(args[9], states)
    torch.testing.assert_close(args[11], torch.tensor([2, 7], dtype=torch.int32))


@pytest.mark.parametrize("padding_value", [77.0, float("nan")], ids=["finite", "nonfinite"])
def test_gdn_adaptive_packed_padding_does_not_enter_live_prefixes(
    monkeypatch: pytest.MonkeyPatch, padding_value: float
) -> None:
    layer = NpuQwen3_5GatedDeltaNet(
        make_linear_config(
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
        ),
        0,
        torch.float32,
        torch.device("cpu"),
    )

    def project(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return hidden, hidden[:, :1], hidden[:, :1], hidden.unsqueeze(1)

    dense_inputs: list[torch.Tensor] = []

    def verify(qkv: torch.Tensor, z: torch.Tensor, *args: object) -> torch.Tensor:
        dense_inputs.append(qkv.clone())
        return z.cumsum(dim=1)

    monkeypatch.setattr(layer, "_project_prefill_inputs", project)
    monkeypatch.setattr(kernels, "mega_gdn_mtp_decode", verify, raising=False)
    hidden = torch.arange(1, 9, dtype=torch.float32).view(8, 1).expand(8, 128).clone()
    hidden[5:].fill_(padding_value)
    metadata = GdnVerifyMetadata(
        state_caches={0: SimpleNamespace(conv=torch.empty(0), ssm=torch.empty(0))},
        state_indices=torch.tensor([1, 2], dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        query_width=4,
        packed_to_dense=torch.tensor([0, 1, 4, 5, 6, 8, 8, 8]),
        dense_to_packed=torch.tensor([0, 1, 8, 8, 2, 3, 4, 8]),
    )

    actual = layer._verify(hidden, metadata)

    expected = torch.tensor([1, 3, 3, 7, 12, 0, 0, 0], dtype=torch.float32).view(8, 1).expand(8, 128)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        dense_inputs[0][:, :, 0], torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0]], dtype=torch.float32)
    )


@pytest.mark.parametrize("key_heads,value_heads", [(1, 1), (4, 12)], ids=["minimal", "qwen38_tp4"])
def test_npu_verify_rejection_matches_sequential_decode(key_heads: int, value_heads: int) -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    from xllm.python.kernels_npu.gated_delta_net import mega_gdn_decode, mega_gdn_mtp_decode

    torch.manual_seed(23)
    device = torch.device("npu:0")
    batch, width, slots = 2, 8, 3
    conv_dim = (2 * key_heads + value_heads) * 128

    def random(shape: tuple[int, ...], scale: float = 0.1) -> torch.Tensor:
        return (torch.randn(shape) * scale).to(device=device, dtype=torch.bfloat16)

    conv_weight = random((4, conv_dim))
    a_log = torch.zeros(value_heads, device=device)
    dt_bias = torch.zeros(value_heads, device=device)
    norm_weight = torch.ones(128, device=device, dtype=torch.bfloat16)
    indices = torch.tensor([1, 2], device=device, dtype=torch.int32)
    conv = torch.zeros(slots, width + 2, conv_dim, device=device, dtype=torch.bfloat16)
    ssm = torch.zeros(slots * width, value_heads, 128, 128, device=device)
    reference_conv = torch.zeros(slots, 3, conv_dim, device=device, dtype=torch.bfloat16)
    reference_ssm = torch.zeros(slots, value_heads, 128, 128, device=device)
    counts = torch.ones(batch, device=device, dtype=torch.int32)
    checkpoints: list[tuple[torch.Tensor, torch.Tensor]] = []
    for iteration in range(2):
        if iteration:
            # Resume after different accepted prefixes of the preceding block.
            for row, accepted in enumerate((1, 5)):
                reference_conv[row + 1].copy_(checkpoints[accepted - 1][0][row + 1])
                reference_ssm[row + 1].copy_(checkpoints[accepted - 1][1][row + 1])
            counts.copy_(torch.tensor([1, 5], device=device, dtype=torch.int32))
        qkv = random((batch, width, conv_dim))
        z = random((batch, width, value_heads, 128))
        b = random((batch, width, value_heads))
        a = random((batch, width, value_heads))
        actual = mega_gdn_mtp_decode(
            qkv, z, b, a, conv_weight, conv, a_log, dt_bias, ssm, indices, indices, counts, norm_weight
        )
        checkpoints = []
        reference_outputs: list[torch.Tensor] = []
        for position in range(width):
            reference_outputs.append(
                mega_gdn_decode(
                    qkv[:, position].contiguous(),
                    z[:, position].contiguous(),
                    b[:, position].contiguous(),
                    a[:, position].contiguous(),
                    conv_weight,
                    reference_conv,
                    a_log,
                    dt_bias,
                    reference_ssm,
                    indices,
                    indices,
                    norm_weight,
                )
            )
            checkpoints.append((reference_conv.clone(), reference_ssm.clone()))
        expected = torch.stack(reference_outputs, dim=1)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@torch.inference_mode()
def test_npu_packed_gdn_graph_replays_ragged_prefixes_after_rejection() -> None:
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    from xllm.python.kernels_npu.gated_delta_net import mega_gdn_decode

    torch.manual_seed(37)
    device = torch.device("npu:0")
    cfg = make_linear_config(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    layer = NpuQwen3_5GatedDeltaNet(cfg, 0, torch.bfloat16, device)
    for parameter in layer.parameters():
        parameter.normal_(std=0.1)
    layer.norm_weight.fill_(1)
    conv = torch.zeros(4, 10, 384, device=device, dtype=torch.bfloat16)
    ssm = torch.zeros(32, 1, 128, 128, device=device)
    reference_conv = torch.zeros(4, 3, 384, device=device, dtype=torch.bfloat16)
    reference_ssm = torch.zeros(4, 1, 128, 128, device=device)
    hidden = torch.zeros(8, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    graph_inputs = LinearVerifyGraphInputs(
        torch.zeros(3, device=device, dtype=torch.int32),
        torch.ones(3, device=device, dtype=torch.int32),
        8,
        torch.full((8,), 24, device=device, dtype=torch.int64),
        torch.full((24,), 8, device=device, dtype=torch.int64),
    )
    inputs = DSparkAdaptiveGraphInputs(
        torch.zeros(264, device=device, dtype=torch.uint8),
        torch.tensor([2 | (1 << 32)], device=device, dtype=torch.int64),
        torch.ones(3, device=device, dtype=torch.int64),
        torch.zeros(3, device=device, dtype=torch.int32),
        torch.ones(3, device=device, dtype=torch.int32),
        torch.zeros(3, 1, device=device, dtype=torch.int32),
        torch.tensor([0, 2, 5, -99], device=device, dtype=torch.int32),
        torch.tensor([0, 0, 1, 1, 1, -1, -1, -1], device=device, dtype=torch.int32),
        torch.tensor([0, 1, 0, 1, 2, -1, -1, -1], device=device, dtype=torch.int32),
        128,
        linear_state_indices=torch.tensor([1, 2, 0], device=device, dtype=torch.int32),
        num_accepted_tokens=torch.ones(3, device=device, dtype=torch.int32),
    )
    metadata = GdnVerifyMetadata(
        {0: SimpleNamespace(conv=conv, ssm=ssm)},
        graph_inputs.state_indices,
        graph_inputs.num_accepted_tokens,
        8,
        graph_inputs.packed_to_dense,
        graph_inputs.dense_to_packed,
    )

    def forward() -> torch.Tensor:
        PreparedAclGraphRunner._refresh_linear_verify_graph(inputs, graph_inputs, 8)
        return layer._verify(hidden, metadata)

    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream(device))
    with torch.npu.stream(stream):
        forward()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            actual = forward()
        checkpoints: list[tuple[torch.Tensor, torch.Tensor]] = []
        for iteration, widths in enumerate(((2, 3), (4, 1))):
            if iteration:
                for row, accepted in enumerate((1, 2)):
                    reference_conv[row + 1].copy_(checkpoints[accepted - 1][0][row + 1])
                    reference_ssm[row + 1].copy_(checkpoints[accepted - 1][1][row + 1])
                inputs.num_accepted_tokens.copy_(torch.tensor([1, 2, 1], device=device, dtype=torch.int32))
            total = sum(widths)
            packed = (torch.randn(8, cfg.hidden_size, device=device) * 0.1).to(torch.bfloat16)
            packed[total:].fill_(float("nan"))
            hidden.copy_(packed)
            offsets = (0, widths[0], total)
            inputs.offsets[:3].copy_(torch.tensor(offsets, device=device, dtype=torch.int32))
            inputs.owners.fill_(-1)
            inputs.columns.fill_(-1)
            for row, width in enumerate(widths):
                inputs.owners[offsets[row] : offsets[row + 1]].fill_(row)
                inputs.columns[offsets[row] : offsets[row + 1]].copy_(torch.arange(width, device=device))
            inputs.control.fill_(2)
            graph.replay()

            qkv, a, b, z = layer._project_prefill_inputs(hidden)
            checkpoints = []
            expected = torch.zeros(8, 128, device=device, dtype=torch.bfloat16)
            indices = torch.tensor([1, 2], device=device, dtype=torch.int32)
            for step in range(max(widths)):
                selected = torch.tensor(
                    [offsets[row] + min(step, width - 1) for row, width in enumerate(widths)], device=device
                )
                output = mega_gdn_decode(
                    qkv.index_select(0, selected),
                    z.index_select(0, selected),
                    b.index_select(0, selected),
                    a.index_select(0, selected),
                    layer.conv1d_weight,
                    reference_conv,
                    layer.A_log,
                    layer.dt_bias,
                    reference_ssm,
                    indices,
                    indices,
                    layer.norm_weight,
                )
                for row, width in enumerate(widths):
                    if step < width:
                        expected[offsets[row] + step].copy_(output[row].reshape(-1))
                checkpoints.append((reference_conv.clone(), reference_ssm.clone()))
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
