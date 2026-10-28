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

"""Explicit paged-KV masks for Speculators block drafts."""

from __future__ import annotations

import torch


def build_block_draft_mask(
    kv_lengths: torch.Tensor,
    block_width: int,
    kv_capacity: int,
    context_window: int,
    causal: bool,
) -> torch.Tensor:
    """Return a B1QK exclusion mask without reading lengths back to Host.

    The final ``block_width`` keys belong to the synthetic draft block.
    Every row sees the same context window before that block's anchor.
    ``causal`` restricts only visibility inside the synthetic block.
    """
    if kv_lengths.ndim != 1 or min(block_width, kv_capacity, context_window) <= 0:
        raise ValueError("block masks require one-dimensional lengths and positive geometry")
    lengths = kv_lengths.view(-1, 1, 1, 1)
    anchors = lengths - block_width
    keys = torch.arange(kv_capacity, device=kv_lengths.device).view(1, 1, 1, -1)
    masked = (keys >= lengths) | (keys < anchors - context_window)
    if causal:
        rows = torch.arange(block_width, device=kv_lengths.device).view(1, 1, -1, 1)
        masked = masked | (keys > anchors + rows)
    return masked.expand(-1, 1, block_width, -1).contiguous()
