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

from __future__ import annotations

import pytest
import torch

from xllm.python.attention.block_draft_mask import build_block_draft_mask


@pytest.mark.parametrize("width", [1, 8])
@pytest.mark.parametrize("causal", [False, True])
def test_block_mask_matches_speculators_visibility(width: int, causal: bool) -> None:
    anchors = [0, 2047, 2048, 2049, 8192]
    capacity = max(anchors) + width + 7
    lengths = torch.tensor([anchor + width for anchor in anchors], dtype=torch.int32)
    mask = build_block_draft_mask(lengths, width, capacity, 2048, causal)
    expected = torch.ones(len(anchors), 1, width, capacity, dtype=torch.bool)
    for batch, anchor in enumerate(anchors):
        expected[batch, 0, :, max(0, anchor - 2048) : anchor] = False
        for row in range(width):
            block_end = anchor + row + 1 if causal else anchor + width
            expected[batch, 0, row, anchor:block_end] = False
    torch.testing.assert_close(mask, expected)
    assert mask.is_contiguous()


def test_all_rows_keep_the_same_context_window() -> None:
    mask = build_block_draft_mask(torch.tensor([12], dtype=torch.int32), 4, 16, 3, True)
    for row in range(4):
        assert torch.where(~mask[0, 0, row])[0].tolist() == list(range(5, 9 + row))


def test_rejected_tail_and_new_device_lengths_change_the_mask() -> None:
    lengths = torch.tensor([12, 9], dtype=torch.int32)
    before = build_block_draft_mask(lengths, 4, 16, 3, True)
    lengths.copy_(torch.tensor([9, 7], dtype=torch.int32))
    after = build_block_draft_mask(lengths, 4, 16, 3, True)
    assert torch.where(~before[0, 0, 0])[0].tolist() == [5, 6, 7, 8]
    assert torch.where(~after[0, 0, 0])[0].tolist() == [2, 3, 4, 5]
    assert after[0, :, :, 9:].all()
    assert after[1, :, :, 7:].all()
