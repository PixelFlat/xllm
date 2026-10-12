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

"""Exercise xLLM's eager ATB cache after torch_npu loads its ATB plugin."""

import pytest
import torch


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires Ascend NPU")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_repeated_eager_cache_writes_with_torch_npu_loaded(dtype: torch.dtype) -> None:
    # conftest loads torch_npu before xllm_export, reproducing the load order
    # that previously resolved xLLM's cache methods against a smaller object.
    cache_shape = (2, 128, 1, 64)
    key_cache = torch.full(cache_shape, 42.0, dtype=dtype, device="npu:0")
    value_cache = torch.full(cache_shape, 43.0, dtype=dtype, device="npu:0")
    slots = torch.tensor([0, 127, 128, -1], dtype=torch.int32, device="npu:0")
    expected_keys = torch.full(cache_shape, 42.0, dtype=dtype)
    expected_values = torch.full(cache_shape, 43.0, dtype=dtype)
    rows = torch.arange(4 * 64, dtype=torch.float32).reshape(4, 1, 64).div(256).to(dtype)
    for iteration in range(2):
        keys = rows + iteration
        values = -keys
        torch.ops.xllm_ops.reshape_paged_cache(slots, keys.to("npu:0"), values.to("npu:0"), key_cache, value_cache)
        torch.npu.synchronize()
        for row, slot in enumerate((0, 127, 128)):
            expected_keys.view(256, 1, 64)[slot].copy_(keys[row])
            expected_values.view(256, 1, 64)[slot].copy_(values[row])
        torch.testing.assert_close(key_cache.cpu(), expected_keys, rtol=0, atol=0)
        torch.testing.assert_close(value_cache.cpu(), expected_values, rtol=0, atol=0)
