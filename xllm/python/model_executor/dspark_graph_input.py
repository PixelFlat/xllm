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

"""Borrowed fixed-capacity inputs for an adaptive DSpark target graph."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class DSparkAdaptiveGraphInputs:
    """Keep Slot-owned storage alive without reading device control on Host.

    ``arena`` has ``12 * draft_width + 4`` bytes per request-capacity row.
    Its live compact proposal,
    confidence and width regions depend on the low 32 bits of ``control[0]``.
    The high 32 bits suppress cache writes during warmup and profiling. Tensor
    layouts stay fixed when the live request count changes between replays.
    """

    arena: torch.Tensor
    control: torch.Tensor
    anchors: torch.Tensor
    base_positions: torch.Tensor
    base_kv_lengths: torch.Tensor
    base_block_table: torch.Tensor
    offsets: torch.Tensor
    owners: torch.Tensor
    columns: torch.Tensor
    block_size: int
    draft_width: int = 7
    # Hybrid targets restore request-scoped checkpoints before packed verify.
    # The preceding accepted lengths are gathered on the ordered task stream.
    linear_state_indices: torch.Tensor | None = None
    num_accepted_tokens: torch.Tensor | None = None
