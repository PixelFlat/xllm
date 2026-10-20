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

"""Device-only reference for ragged greedy output and context KV commitment.

This defines the dedicated kernel's contract; it is not a serving fallback.
The prefix allocator/packer guarantees positive widths, consistent offsets
and valid owners. No device value selects a host output shape or launch.
"""

from __future__ import annotations

import torch


def _check_tensor(
    name: str, tensor: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> None:
    if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous with shape={shape}, dtype={dtype}, device={device}")


def greedy_verify_commit_out(
    proposals: torch.Tensor,
    packed_target_ids: torch.Tensor,
    widths: torch.Tensor,
    offsets: torch.Tensor,
    owners: torch.Tensor,
    columns: torch.Tensor,
    source_slots: torch.Tensor,
    tokens: torch.Tensor,
    lengths: torch.Tensor,
    context_slots: torch.Tensor,
) -> None:
    """Write a corrected greedy prefix and context slots into fixed buffers.

    Width includes one anchor. Every output prefix includes the first target
    correction, or the request's last packed target ID as its full-acceptance
    bonus. Context rows include the anchor and accepted proposals, not the
    correction/bonus token, which has not been forwarded through the target.
    Inputs must already have TP consensus; the target is greedy without
    requested log-probabilities. All buffers must be non-overlapping.
    """
    if proposals.ndim != 2 or proposals.shape[1] != 7 or packed_target_ids.ndim != 1:
        raise ValueError("proposals must be [batch,7] and packed target IDs one-dimensional")
    batch_size = proposals.shape[0]
    capacity = packed_target_ids.shape[0]
    if capacity < batch_size:
        raise ValueError("packed capacity must contain at least one anchor per request")
    device = proposals.device
    inputs = (
        ("proposals", proposals, (batch_size, 7), torch.int64),
        ("target IDs", packed_target_ids, (capacity,), torch.int64),
        ("widths", widths, (batch_size,), torch.int32),
        ("offsets", offsets, (batch_size + 1,), torch.int32),
        ("owners", owners, (capacity,), torch.int32),
        ("columns", columns, (capacity,), torch.int32),
        ("source slots", source_slots, (capacity,), torch.int32),
    )
    outputs = (
        ("tokens", tokens, (batch_size, 8), torch.int64),
        ("lengths", lengths, (batch_size,), torch.int32),
        ("context slots", context_slots, (capacity,), torch.int32),
    )
    for name, tensor, shape, dtype in (*inputs, *outputs):
        _check_tensor(name, tensor, shape, dtype, device)
    for output_index, (_, output, _, _) in enumerate(outputs):
        start = output.data_ptr()
        end = start + output.numel() * output.element_size()
        for _, other, _, _ in (*inputs, *outputs[:output_index]):
            other_start = other.data_ptr()
            other_end = other_start + other.numel() * other.element_size()
            if max(start, other_start) < min(end, other_end):
                raise ValueError("greedy commit buffers must not overlap")
    if batch_size == 0:
        context_slots.fill_(-1)
        return
    column_ids = torch.arange(8, dtype=torch.int32, device=device)
    # Only the small token-ID result is rectangular. Hidden states and logits
    # remain packed; each request loads its own bounded target rows.
    target_indices = offsets[:-1, None] + torch.minimum(column_ids[None, :], widths[:, None] - 1)
    target_tokens = packed_target_ids.index_select(0, target_indices.flatten().to(torch.int64)).view(batch_size, 8)
    compared = column_ids[None, :7] < widths[:, None] - 1
    rejected = compared & (proposals != target_tokens[:, :7])
    first_rejected = torch.where(rejected, column_ids[None, :7], 7).amin(dim=1)
    committed_lengths = torch.minimum(first_rejected + 1, widths)
    lengths.copy_(committed_lengths)
    tokens.copy_(target_tokens)
    tokens.masked_fill_(column_ids[None, :] >= committed_lengths[:, None], -1)
    owner_lengths = committed_lengths.index_select(0, owners.clamp_min(0).to(torch.int64))
    valid_context = (owners >= 0) & (columns >= 0) & (columns < owner_lengths)
    context_slots.copy_(source_slots)
    context_slots.masked_fill_(~valid_context, -1)
