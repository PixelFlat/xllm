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

"""Check the test-only HCCL port defaults before communicator initialization."""

from __future__ import annotations

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.python import nz_test_utils


@pytest.mark.parametrize("port_range", [None, "auto", "57000", "61000-61050", ""])
def test_init_hccl_uses_auto_unless_port_range_is_configured(
    monkeypatch: pytest.MonkeyPatch, port_range: str | None
) -> None:
    if port_range is None:
        monkeypatch.delenv("HCCL_NPU_SOCKET_PORT_RANGE", raising=False)
    else:
        monkeypatch.setenv("HCCL_NPU_SOCKET_PORT_RANGE", port_range)
    expected_ports = "auto" if port_range is None else port_range
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    npu_config = SimpleNamespace(allow_internal_format=False)
    monkeypatch.setattr(torch.npu, "config", npu_config)

    def _set_device(device: int) -> None:
        assert os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] == expected_ports
        assert device == 1

    def _init_process_group(*, backend: str, timeout: timedelta, device_id: torch.device) -> None:
        assert os.environ["HCCL_NPU_SOCKET_PORT_RANGE"] == expected_ports
        assert backend == "hccl"
        assert timeout == timedelta(seconds=90)
        assert device_id == torch.device("npu:1")
        assert npu_config.allow_internal_format

    set_device = Mock(side_effect=_set_device)
    init_process_group = Mock(side_effect=_init_process_group)
    monkeypatch.setattr(torch.npu, "set_device", set_device)
    monkeypatch.setattr(nz_test_utils.dist, "init_process_group", init_process_group)

    assert nz_test_utils.init_hccl(timeout_s=90) == (1, 2, 1)
    set_device.assert_called_once_with(1)
    init_process_group.assert_called_once_with(
        backend="hccl", timeout=timedelta(seconds=90), device_id=torch.device("npu:1")
    )
