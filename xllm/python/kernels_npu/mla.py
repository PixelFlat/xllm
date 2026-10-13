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

"""Capturable DeepSeek MLA preprocessing regions for Ascend NPU."""

from __future__ import annotations

import struct

import torch
import torch_npu

from xllm.python.model_executor.forward_context import get_forward_context_or_none

from .linear import atb_matmul_ein_sum
from .normalization import rms_norm, rms_norm_dynamic_quant
from .quantization import dynamic_quant, quant_matmul, quant_matmul_out, quantize_per_tensor

_KROPE_CTKV_CACHE_MODE = 1
_NZ_CACHE_MODE = 3
_PER_TENSOR_QUANT_ASYMM_MODE = 0
MLA_PREPROCESS_V2_MAX_TOKENS = 1024
_MLA_PREPROCESS_V2_KV_LORA_RANK = 512
_MLA_PREPROCESS_V2_QK_ROPE_HEAD_DIM = 64
_KV_RMSNORM_ROPE_CACHE = getattr(
    torch_npu,
    "npu_kv_rmsnorm_rope_cache",
    None,
)
_STRIDED_Q_A_QUANT_OUT = getattr(torch.ops.xllm_ops, "strided_q_a_rms_norm_dynamic_quant_out", None)
_STRIDED_MLA_KV_CACHE_OUT = getattr(torch.ops.xllm_ops, "strided_mla_kv_cache_out", None)
# ModelArgs stores epsilon as a C++ float before exporting it to Python.
# Accept only the literal and its exact FP32 representation, not nearby values.
_STRIDED_Q_A_QUANT_EPSILONS = (1e-5, struct.unpack("f", struct.pack("f", 1e-5))[0])


def supports_strided_q_a_quant_out(epsilon: float = 1e-5) -> bool:
    """Whether the initialized native library provides direct Q-A quantization."""
    return _STRIDED_Q_A_QUANT_OUT is not None and epsilon in _STRIDED_Q_A_QUANT_EPSILONS


def supports_strided_mla_kv_cache_out(epsilon: float = 1e-5) -> bool:
    """Whether the initialized library provides the fixed-epsilon KV writer."""
    return _STRIDED_MLA_KV_CACHE_OUT is not None and epsilon in _STRIDED_Q_A_QUANT_EPSILONS


def supports_strided_mla_kv_cache_layout(kv_cache: torch.Tensor, rope_cache: torch.Tensor) -> bool:
    """Select the direct writer only for its supported one-head PA storage."""
    return (
        kv_cache.device.type in ("npu", "privateuseone")
        and kv_cache.device == rope_cache.device
        and kv_cache.dtype == rope_cache.dtype == torch.bfloat16
        and kv_cache.ndim == rope_cache.ndim
        and kv_cache.ndim in (3, 4)
        and kv_cache.shape[:-1] == rope_cache.shape[:-1]
        and kv_cache.shape[-1] == 512
        and rope_cache.shape[-1] == 64
        and (kv_cache.ndim == 3 or kv_cache.shape[2] == 1)
        and 0 < kv_cache.numel() <= 2**31 - 1
        and kv_cache.is_contiguous()
        and rope_cache.is_contiguous()
        and torch_npu.get_npu_format(kv_cache) in (torch_npu.Format.ND, torch_npu.Format.NCHW)
        and torch_npu.get_npu_format(rope_cache) in (torch_npu.Format.ND, torch_npu.Format.NCHW)
    )


def _validate_strided_mla_kv_cache_geometry(
    tokens: int, kv_lora_rank: int, q_lora_rank: int, qk_rope_head_dim: int, epsilon: float
) -> None:
    if (
        tokens not in (1, 2, 4, 8, 16)
        or kv_lora_rank != 512
        or q_lora_rank != 2048
        or qk_rope_head_dim != 64
        or epsilon not in _STRIDED_Q_A_QUANT_EPSILONS
    ):
        raise ValueError("Strided KV requires supported decode graph geometry")


def _validate_strided_q_a_quant_geometry(
    tokens: int,
    kv_lora_rank: int,
    q_lora_rank: int,
    qk_rope_head_dim: int,
    epsilon: float,
    fuse_q_norm_quant: bool,
) -> None:
    if (
        not fuse_q_norm_quant
        or tokens not in (1, 2, 4, 8, 16)
        or kv_lora_rank != 512
        or qk_rope_head_dim != 64
        or q_lora_rank != 2048
        or epsilon not in _STRIDED_Q_A_QUANT_EPSILONS
    ):
        raise ValueError("Strided Q-A requires supported fused decode graph geometry")


def _quantize_strided_q_a(
    projection: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Write Q-A without materializing its view or retaining intermediates."""
    tokens = projection.shape[0]
    # Capture owns the allocation's replay address; consumers own its lifetime.
    output = torch.empty((tokens, 2048), dtype=torch.int8, device=projection.device)
    scales = torch.empty((tokens,), dtype=torch.float32, device=projection.device)
    _STRIDED_Q_A_QUANT_OUT(projection, norm_weight, output, scales)
    return output, scales


def has_mla_preprocess_v2() -> bool:
    """Return whether the xLLM binding and CANN MLAPO v2 symbols exist."""
    capability_op = getattr(torch.ops.xllm_ops, "has_mla_preprocess_v2", None)
    return capability_op is not None and bool(capability_op())


def supports_mla_preprocess_v2(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> bool:
    """Return whether MLAPO v2 supports this MLA layout."""
    return (
        kv_lora_rank == _MLA_PREPROCESS_V2_KV_LORA_RANK
        and qk_rope_head_dim == _MLA_PREPROCESS_V2_QK_ROPE_HEAD_DIM
        and has_mla_preprocess_v2()
    )


def _reorder_rope_axis(
    tensor: torch.Tensor,
    rope_dim: int,
    dim: int,
) -> torch.Tensor:
    """Move even RoPE channels before odd channels on one tensor axis."""
    axis_size = tensor.shape[dim]
    rope_start = axis_size - rope_dim
    prefix = tensor.narrow(dim, 0, rope_start)
    rope = tensor.narrow(dim, rope_start, rope_dim)
    even = rope.index_select(
        dim,
        torch.arange(0, rope_dim, 2, device=tensor.device),
    )
    odd = rope.index_select(
        dim,
        torch.arange(1, rope_dim, 2, device=tensor.device),
    )
    return torch.cat((prefix, even, odd), dim=dim).contiguous()


def prepare_mla_preprocess_v2_qkv(
    weight: torch.Tensor,
    descale: torch.Tensor,
    bias: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare fused KV-A/Q-A tensors for ``MlaPreprocessV2``."""
    kv_dim = kv_lora_rank + qk_rope_head_dim

    def _prepare(tensor: torch.Tensor) -> torch.Tensor:
        kv = _reorder_rope_axis(tensor.narrow(0, 0, kv_dim), qk_rope_head_dim, 0)
        q = tensor.narrow(0, kv_dim, tensor.shape[0] - kv_dim)
        return torch.cat((kv, q), dim=0).contiguous()

    prepared_weight = _prepare(weight)
    if prepared_weight.device.type != "cpu":
        prepared_weight = torch_npu.npu_format_cast(prepared_weight, torch_npu.Format.FRACTAL_NZ)
    return prepared_weight, _prepare(descale), _prepare(bias)


def prepare_mla_preprocess_v2_q_b(
    weight: torch.Tensor,
    descale: torch.Tensor,
    bias: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare Q-B tensors for ``MlaPreprocessV2``."""
    head_dim = qk_nope_head_dim + qk_rope_head_dim

    def _prepare(tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape
        viewed = tensor.view(num_heads, head_dim, *shape[1:])
        reordered = _reorder_rope_axis(viewed, qk_rope_head_dim, 1)
        return reordered.reshape(shape).contiguous()

    # Let the format cast pack NZ storage while preserving the logical matrix.
    prepared_weight = _prepare(weight)
    if prepared_weight.device.type != "cpu":
        prepared_weight = torch_npu.npu_format_cast(prepared_weight, torch_npu.Format.FRACTAL_NZ)
    return prepared_weight, _prepare(descale), _prepare(bias)


def deepseek_mla_preprocess_decode_v2(
    hidden: torch.Tensor,
    input_norm_weight: torch.Tensor,
    input_norm_bias: torch.Tensor,
    qkv_input_scale: torch.Tensor,
    qkv_input_offset: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_deq_scale: torch.Tensor,
    qkv_quant_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_norm_bias: torch.Tensor,
    q_b_input_scale: torch.Tensor,
    q_b_input_offset: torch.Tensor,
    q_b_weight: torch.Tensor,
    q_b_deq_scale: torch.Tensor,
    q_b_quant_bias: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    w_uk: torch.Tensor,
    kv_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_lora_rank: int,
    q_lora_rank: int,
    qk_rope_head_dim: int,
    norm_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused xLLM MLAPO v2 decode path."""
    cache_mode = _NZ_CACHE_MODE if _mla_cache_mode(kv_cache) == "PA_NZ" else _KROPE_CTKV_CACHE_MODE
    num_tokens = hidden.shape[0]
    q_latent, _, q_pe, _, q_c = torch.ops.xllm_ops.mla_preprocess_v2(
        hidden,
        input_norm_weight,
        input_norm_bias,
        qkv_input_scale,
        qkv_input_offset,
        qkv_weight,
        qkv_deq_scale,
        qkv_quant_bias,
        q_norm_weight,
        q_norm_bias,
        q_b_input_scale,
        q_b_input_offset,
        q_b_weight,
        q_b_deq_scale,
        q_b_quant_bias,
        kv_norm_weight,
        rope_cos.view(num_tokens, -1),
        rope_sin.view(num_tokens, -1),
        w_uk,
        kv_cache,
        rope_cache,
        slot_mapping,
        qkv_input_scale,
        q_b_input_scale,
        q_lora_rank,
        qk_rope_head_dim,
        qk_rope_head_dim,
        norm_epsilon,
        2,
        2,
        True,
        True,
        True,
        cache_mode,
        _PER_TENSOR_QUANT_ASYMM_MODE,
        False,
        1,
        True,
    )
    return q_c, q_latent, q_pe


def _mla_cache_mode(kv_cache: torch.Tensor) -> str:
    """Return the cache layout name expected by the fused KV operator."""
    get_npu_format = getattr(torch_npu, "get_npu_format", None)
    if get_npu_format is not None and get_npu_format(kv_cache) == torch_npu.Format.FRACTAL_NZ:
        return "PA_NZ"
    return "PA"


def supports_mla_kv_cache_slot_reuse(kv_cache: torch.Tensor) -> bool:
    """Whether this cache writer consumes the model's shared INT64 slots."""
    return _KV_RMSNORM_ROPE_CACHE is not None and _mla_cache_mode(kv_cache) == "PA"


def _write_mla_kv_cache(
    kv: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    kv_norm_epsilon: float,
    slot_mapping_int64: torch.Tensor | None = None,
) -> None:
    """Normalize and rotate MLA KV, then write it into the paged cache."""
    num_tokens = kv.shape[0]
    cache_mode = _mla_cache_mode(kv_cache)
    # xLLM's PA_NZ cache is an internal FRACTAL_NZ tensor. The torch-npu
    # operator currently accepts PA_NZ only as a physically packed ND tensor,
    # so keep the established cache writer for the internal-format variant.
    if _KV_RMSNORM_ROPE_CACHE is not None and cache_mode == "PA":
        if slot_mapping_int64 is not None and (
            slot_mapping_int64.dtype != torch.int64
            or slot_mapping_int64.shape != slot_mapping.shape
            or slot_mapping_int64.device != slot_mapping.device
            or not slot_mapping_int64.is_contiguous()
        ):
            raise ValueError("prepared MLA slots must be contiguous INT64 matching the original slot mapping")
        kv_no_split = kv.view(
            num_tokens,
            1,
            1,
            kv_lora_rank + qk_rope_head_dim,
        )
        _KV_RMSNORM_ROPE_CACHE(
            kv_no_split,
            kv_norm_weight,
            rope_cos,
            rope_sin,
            slot_mapping.to(torch.int64) if slot_mapping_int64 is None else slot_mapping_int64,
            rope_cache,
            kv_cache,
            epsilon=kv_norm_epsilon,
            cache_mode=cache_mode,
        )
        return

    k_latent_raw, k_rope_raw = kv.split([kv_lora_rank, qk_rope_head_dim], dim=-1)
    k_latent = rms_norm(k_latent_raw, kv_norm_weight, kv_norm_epsilon).view(num_tokens, 1, kv_lora_rank)
    k_pe = torch_npu.npu_interleave_rope(
        k_rope_raw.view(num_tokens, 1, 1, qk_rope_head_dim),
        rope_cos,
        rope_sin,
    ).view(num_tokens, 1, qk_rope_head_dim)
    torch.ops.xllm_ops.reshape_paged_cache(
        slot_mapping,
        k_latent,
        k_pe,
        kv_cache,
        rope_cache,
    )


@torch.library.custom_op(
    "xllm_python::deepseek_mla_preprocess_decode",
    mutates_args={"kv_cache", "rope_cache"},
)
def deepseek_mla_preprocess_decode(
    hidden: torch.Tensor,
    qkv_input_scale: torch.Tensor,
    qkv_input_offset: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_deq_scale: torch.Tensor,
    qkv_quant_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_b_input_scale: torch.Tensor,
    q_b_input_offset: torch.Tensor,
    q_b_weight: torch.Tensor,
    q_b_deq_scale: torch.Tensor,
    q_b_quant_bias: torch.Tensor,
    w_uk: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    kv_lora_rank: int,
    q_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    q_norm_epsilon: float,
    kv_norm_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the complete capturable DeepSeek decode MLA preprocessing path."""
    hidden_int8 = quantize_per_tensor(
        hidden,
        qkv_input_scale,
        qkv_input_offset,
        torch.qint8,
        -1,
    )
    qkv_a = quant_matmul(
        hidden_int8,
        qkv_weight,
        False,
        qkv_deq_scale,
        None,
        None,
        qkv_quant_bias,
        torch.bfloat16,
    )
    kv_dim = kv_lora_rank + qk_rope_head_dim
    kv, q_a = qkv_a.split([kv_dim, q_lora_rank], dim=-1)
    q_c = rms_norm(q_a, q_norm_weight, q_norm_epsilon)
    q_c_quant = quantize_per_tensor(
        q_c,
        q_b_input_scale,
        q_b_input_offset,
        torch.qint8,
        -1,
    )
    q = quant_matmul(
        q_c_quant,
        q_b_weight,
        False,
        q_b_deq_scale,
        None,
        None,
        q_b_quant_bias,
        torch.bfloat16,
    ).view(
        hidden.shape[0],
        num_heads,
        qk_nope_head_dim + qk_rope_head_dim,
    )
    q_nope, q_rope = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    q_latent = atb_matmul_ein_sum(q_nope, w_uk)
    num_tokens = hidden.shape[0]
    q_pe = torch_npu.npu_interleave_rope(
        q_rope.view(num_tokens, num_heads, 1, qk_rope_head_dim),
        rope_cos,
        rope_sin,
    ).view(num_tokens, num_heads, qk_rope_head_dim)
    _write_mla_kv_cache(
        kv,
        kv_norm_weight,
        rope_cos,
        rope_sin,
        slot_mapping,
        kv_cache,
        rope_cache,
        kv_lora_rank,
        qk_rope_head_dim,
        kv_norm_epsilon,
    )
    return q_c, q_latent, q_pe


def _validate_decode_latent_weight(
    weight: torch.Tensor,
    num_heads: int,
    nope_dim: int,
    rope_dim: int,
    latent_dim: int,
) -> None:
    if (
        weight.ndim != 3
        or weight.shape[0] != num_heads
        or weight.shape[1] not in (nope_dim, nope_dim + rope_dim)
        or weight.shape[2] != latent_dim
    ):
        raise ValueError("decode latent weight must be [heads, nope or full query dim, latent dim]")


def _project_decode_latent_query(
    query: torch.Tensor,
    weight: torch.Tensor,
    nope_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.shape[1] == nope_dim + rope_dim:
        query_rope = query.narrow(-1, nope_dim, rope_dim)
        query_latent = atb_matmul_ein_sum(query, weight)
    else:
        query_nope, query_rope = query.split([nope_dim, rope_dim], dim=-1)
        query_latent = atb_matmul_ein_sum(query_nope, weight)
    return query_latent, query_rope


@torch.library.custom_op(
    "xllm_python::deepseek_mla_preprocess_decode_dynamic",
    mutates_args={"kv_cache", "rope_cache"},
)
def deepseek_mla_preprocess_decode_dynamic(
    hidden: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_weight_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_b_weight: torch.Tensor,
    q_b_weight_scale: torch.Tensor,
    w_uk: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    kv_lora_rank: int,
    q_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    q_norm_epsilon: float,
    kv_norm_epsilon: float,
    fuse_q_norm_quant: bool,
    hidden_scale: torch.Tensor | None = None,
    slot_mapping_int64: torch.Tensor | None = None,
    query_rope_patterns: torch.Tensor | None = None,
    strided_q_a_quant: bool = False,
    kv_rope_patterns: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess MLA and return Q-A and the activation scale used by Q-B.

    Fusion is enabled only when the indexer can consume the same INT8 Q-A as
    Q-B; otherwise it receives the normalized floating-point activation.
    """
    _validate_decode_latent_weight(w_uk, num_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank)
    forward_context = get_forward_context_or_none()
    graph_buffers = forward_context.execution_state if forward_context is not None else None
    if strided_q_a_quant:
        _validate_strided_q_a_quant_geometry(
            hidden.shape[0], kv_lora_rank, q_lora_rank, qk_rope_head_dim, q_norm_epsilon, fuse_q_norm_quant
        )
        if graph_buffers is None:
            raise ValueError("Strided Q-A requires a graph-owned execution state")
    if kv_rope_patterns is not None:
        _validate_strided_mla_kv_cache_geometry(
            hidden.shape[0], kv_lora_rank, q_lora_rank, qk_rope_head_dim, kv_norm_epsilon
        )
        if graph_buffers is None:
            raise ValueError("Strided KV requires a graph-owned execution state")
        if not supports_strided_mla_kv_cache_layout(kv_cache, rope_cache):
            raise ValueError("Strided KV requires contiguous one-head BF16 PA caches")
        if slot_mapping_int64 is not None and (
            slot_mapping_int64.dtype != torch.int64
            or slot_mapping_int64.shape != slot_mapping.shape
            or slot_mapping_int64.device != slot_mapping.device
            or not slot_mapping_int64.is_contiguous()
        ):
            raise ValueError("prepared MLA slots must be contiguous INT64 matching the original slot mapping")

    qkv_output = None
    q_output = None
    if graph_buffers is not None:
        token_count = hidden.shape[0]
        qkv_output = torch.empty(
            (token_count, qkv_weight.shape[1]),
            dtype=torch.bfloat16,
            device=hidden.device,
        )
        q_output = torch.empty(
            (token_count, q_b_weight.shape[1]),
            dtype=torch.bfloat16,
            device=hidden.device,
        )

    if hidden_scale is None:
        hidden_int8, hidden_scale = dynamic_quant(hidden)
    else:
        hidden_int8 = hidden
        hidden_scale = hidden_scale.reshape(-1)
    if qkv_output is None:
        qkv_a = quant_matmul(
            hidden_int8,
            qkv_weight,
            False,
            qkv_weight_scale,
            None,
            hidden_scale,
            None,
            torch.bfloat16,
        )
    else:
        qkv_a = quant_matmul_out(
            hidden_int8,
            qkv_weight,
            False,
            qkv_weight_scale,
            None,
            hidden_scale,
            None,
            torch.bfloat16,
            qkv_output,
        )
    kv_dim = kv_lora_rank + qk_rope_head_dim
    if strided_q_a_quant:
        if kv_rope_patterns is None:
            kv = qkv_a[:, :kv_dim]
        q_c_int8, q_c_scale = _quantize_strided_q_a(qkv_a, q_norm_weight)
        q_c = q_c_int8
    elif fuse_q_norm_quant:
        kv, q_a = qkv_a.split([kv_dim, q_lora_rank], dim=-1)
        q_c_int8, q_c_scale = rms_norm_dynamic_quant(q_a, q_norm_weight, q_norm_epsilon)
        q_c = q_c_int8
    else:
        kv, q_a = qkv_a.split([kv_dim, q_lora_rank], dim=-1)
        q_c = rms_norm(q_a, q_norm_weight, q_norm_epsilon)
        q_c_int8, q_c_scale = dynamic_quant(q_c)
    if q_output is None:
        q = quant_matmul(
            q_c_int8,
            q_b_weight,
            False,
            q_b_weight_scale,
            None,
            q_c_scale,
            None,
            torch.bfloat16,
        )
    else:
        q = quant_matmul_out(
            q_c_int8,
            q_b_weight,
            False,
            q_b_weight_scale,
            None,
            q_c_scale,
            None,
            torch.bfloat16,
            q_output,
        )
    q = q.view(
        hidden.shape[0],
        num_heads,
        qk_nope_head_dim + qk_rope_head_dim,
    )
    q_latent, q_rope = _project_decode_latent_query(q, w_uk, qk_nope_head_dim, qk_rope_head_dim)
    num_tokens = hidden.shape[0]
    if query_rope_patterns is None:
        q_pe = torch_npu.npu_interleave_rope(
            q_rope.view(num_tokens, num_heads, 1, qk_rope_head_dim),
            rope_cos,
            rope_sin,
        ).view(num_tokens, num_heads, qk_rope_head_dim)
    else:
        q_pe = torch.empty((num_tokens, num_heads, qk_rope_head_dim), dtype=q.dtype, device=q.device)
        torch.ops.xllm_ops.mla_query_rope_out(q, rope_cos, rope_sin, query_rope_patterns, q_pe)
    if kv_rope_patterns is not None:
        _STRIDED_MLA_KV_CACHE_OUT(
            qkv_a,
            kv_norm_weight,
            rope_cos,
            rope_sin,
            slot_mapping.to(torch.int64) if slot_mapping_int64 is None else slot_mapping_int64,
            kv_rope_patterns,
            kv_cache,
            rope_cache,
        )
    else:
        _write_mla_kv_cache(
            kv,
            kv_norm_weight,
            rope_cos,
            rope_sin,
            slot_mapping,
            kv_cache,
            rope_cache,
            kv_lora_rank,
            qk_rope_head_dim,
            kv_norm_epsilon,
            slot_mapping_int64,
        )
    return q_c, q_c_scale, q_latent, q_pe


@deepseek_mla_preprocess_decode_dynamic.register_fake
def _deepseek_mla_preprocess_decode_dynamic_fake(
    hidden: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_weight_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_b_weight: torch.Tensor,
    q_b_weight_scale: torch.Tensor,
    w_uk: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    kv_lora_rank: int,
    q_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    q_norm_epsilon: float,
    kv_norm_epsilon: float,
    fuse_q_norm_quant: bool,
    hidden_scale: torch.Tensor | None = None,
    slot_mapping_int64: torch.Tensor | None = None,
    query_rope_patterns: torch.Tensor | None = None,
    strided_q_a_quant: bool = False,
    kv_rope_patterns: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_decode_latent_weight(w_uk, num_heads, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank)
    if strided_q_a_quant:
        _validate_strided_q_a_quant_geometry(
            hidden.shape[0], kv_lora_rank, q_lora_rank, qk_rope_head_dim, q_norm_epsilon, fuse_q_norm_quant
        )
    if kv_rope_patterns is not None:
        _validate_strided_mla_kv_cache_geometry(
            hidden.shape[0], kv_lora_rank, q_lora_rank, qk_rope_head_dim, kv_norm_epsilon
        )
    del (
        qkv_weight,
        qkv_weight_scale,
        q_norm_weight,
        q_b_weight,
        q_b_weight_scale,
        kv_norm_weight,
        rope_cos,
        rope_sin,
        slot_mapping,
        kv_cache,
        rope_cache,
        kv_lora_rank,
        qk_nope_head_dim,
        q_norm_epsilon,
        kv_norm_epsilon,
    )
    num_tokens = hidden.shape[0]
    return (
        hidden.new_empty((num_tokens, q_lora_rank), dtype=torch.int8 if fuse_q_norm_quant else torch.bfloat16),
        hidden.new_empty((num_tokens,), dtype=torch.float32),
        hidden.new_empty((num_tokens, num_heads, w_uk.shape[-1]), dtype=torch.bfloat16),
        hidden.new_empty((num_tokens, num_heads, qk_rope_head_dim), dtype=torch.bfloat16),
    )


@deepseek_mla_preprocess_decode.register_fake
def _deepseek_mla_preprocess_decode_fake(
    hidden: torch.Tensor,
    qkv_input_scale: torch.Tensor,
    qkv_input_offset: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_deq_scale: torch.Tensor,
    qkv_quant_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_b_input_scale: torch.Tensor,
    q_b_input_offset: torch.Tensor,
    q_b_weight: torch.Tensor,
    q_b_deq_scale: torch.Tensor,
    q_b_quant_bias: torch.Tensor,
    w_uk: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    kv_lora_rank: int,
    q_lora_rank: int,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    q_norm_epsilon: float,
    kv_norm_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        qkv_input_scale,
        qkv_input_offset,
        qkv_weight,
        qkv_deq_scale,
        qkv_quant_bias,
        q_norm_weight,
        q_b_input_scale,
        q_b_input_offset,
        q_b_weight,
        q_b_deq_scale,
        q_b_quant_bias,
        kv_norm_weight,
        rope_cos,
        rope_sin,
        slot_mapping,
        kv_cache,
        rope_cache,
        kv_lora_rank,
        qk_nope_head_dim,
        q_norm_epsilon,
        kv_norm_epsilon,
    )
    num_tokens = hidden.shape[0]
    q_c = hidden.new_empty((num_tokens, q_lora_rank))
    q_latent = hidden.new_empty((num_tokens, num_heads, w_uk.shape[-1]))
    q_pe = hidden.new_empty((num_tokens, num_heads, qk_rope_head_dim))
    return q_c, q_latent, q_pe


__all__ = [
    "MLA_PREPROCESS_V2_MAX_TOKENS",
    "deepseek_mla_preprocess_decode",
    "deepseek_mla_preprocess_decode_dynamic",
    "deepseek_mla_preprocess_decode_v2",
    "has_mla_preprocess_v2",
    "prepare_mla_preprocess_v2_q_b",
    "prepare_mla_preprocess_v2_qkv",
    "supports_mla_preprocess_v2",
    "supports_strided_q_a_quant_out",
    "supports_strided_mla_kv_cache_out",
    "supports_strided_mla_kv_cache_layout",
]
