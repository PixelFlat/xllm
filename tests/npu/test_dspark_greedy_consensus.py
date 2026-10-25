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

"""Opt-in two-rank DSpark greedy parity, independent of model/server weights.

Set XLLM_TEST_HCCL_DEVICES=<id>,<id> only after checking those devices are free.
This checks real HCCL, replicated heads and the native prefix allocator, not
the C++ Host budget planner. No payload broadcast repairs the tested outputs.
Seeded dense-head parity is not validation of a trained model checkpoint.
Strict cross-rank comparison reads local raw snapshots after both ranks exit,
without inserting diagnostic collectives into the captured TP communicator.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import pytest

from scripts.logger import logger


def _check_rank_snapshots(artifacts: Path) -> None:
    names = ("hidden", "gathered_logits", "proposals", "confidence", "widths", "dense_confidence", "dense_widths")
    mismatches = []
    for dtype in ("torch.float32", "torch.bfloat16"):
        for batch in (2, 3):
            for label in ("eager", "ties", "near_ties", "changed"):
                paths = [artifacts / f"rank-{rank}-b{batch}-{dtype}-{label}-local.json" for rank in range(2)]
                snapshots = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
                for rank, snapshot in enumerate(snapshots):
                    assert snapshot["rank"] == rank and snapshot["batch_size"] == batch and snapshot["label"] == label
                    assert tuple(output["name"] for output in snapshot["outputs"]) == names, (
                        f"invalid outputs: {paths[rank]}"
                    )
                for left, right in zip(snapshots[0]["outputs"], snapshots[1]["outputs"], strict=True):
                    assert all(left[key] == right[key] for key in ("name", "shape", "dtype")), (
                        f"metadata differs: {paths}"
                    )
                    left_bits, right_bits = left["int32_bits"], right["int32_bits"]
                    assert len(left_bits) == len(right_bits), f"bit count differs: {paths}"
                    if left_bits != right_bits:
                        index = next(
                            index for index, (a, b) in enumerate(zip(left_bits, right_bits, strict=True)) if a != b
                        )
                        mismatches.append(
                            f"{left['name']}: {paths[0].name} vs {paths[1].name}; "
                            f"first word {index}: {left_bits[index]} != {right_bits[index]}"
                        )
    assert len(list(artifacts.glob("*-local.json"))) == 32, f"incomplete local snapshots: {artifacts}"
    assert not mismatches, f"{len(mismatches)} TP bitwise mismatches; inspect {artifacts}:\n" + "\n".join(mismatches)


def _run_workers(tmp_path: Path, *, all_reduce_only: bool = False) -> tuple[Path, list[dict]]:
    selection = os.environ.get("XLLM_TEST_HCCL_DEVICES")
    if selection is None:
        pytest.skip("set XLLM_TEST_HCCL_DEVICES to two verified-free logical NPU ids")
    parts = selection.split(",")
    assert len(parts) == 2 and all(part.strip().isdecimal() for part in parts), "expected two nonnegative NPU ids"
    devices = tuple(int(part) for part in parts)
    assert devices[0] != devices[1], "HCCL ranks require distinct devices"

    parent = Path(os.environ.get("XLLM_TEST_HCCL_ARTIFACT_DIR", str(tmp_path))).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix="dspark-greedy-", dir=parent))
    repo = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    environment["HCCL_OP_EXPANSION_MODE"] = "AIV"
    environment.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    processes: list[subprocess.Popen[bytes]] = []
    timed_out = False
    logger.info("DSpark greedy consensus artifacts: %s", artifacts)
    with ExitStack() as stack:
        try:
            for rank, device in enumerate(devices):
                log = stack.enter_context((artifacts / f"rank-{rank}.log").open("xb"))
                command = [
                    sys.executable,
                    "-m",
                    "tests.npu.dspark_greedy_consensus_worker",
                    "--rank",
                    str(rank),
                    "--device",
                    str(device),
                    "--artifact-dir",
                    str(artifacts),
                ]
                if all_reduce_only:
                    command.append("--all-reduce-only")
                processes.append(
                    subprocess.Popen(
                        command,
                        cwd=repo,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                )
            deadline = time.monotonic() + 180
            while True:
                codes = [process.poll() for process in processes]
                if all(code is not None for code in codes) or any(code not in (None, 0) for code in codes):
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    break
                time.sleep(0.1)
        finally:
            # Scope cleanup to our fresh children; never inspect/stop other jobs.
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            summary = {"devices": devices, "returncodes": [p.returncode for p in processes], "timed_out": timed_out}
            (artifacts / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    assert not timed_out, f"DSpark consensus probe exceeded 180s; inspect {artifacts}"
    assert summary["returncodes"] == [0, 0], f"rank failure; inspect {artifacts}"
    results = [json.loads((artifacts / f"rank-{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    assert [result["rank"] for result in results] == [0, 1]
    worker = Path(__file__).resolve().with_name("dspark_greedy_consensus_worker.py")
    source_hash = hashlib.sha256(worker.read_bytes()).hexdigest()
    assert all(result["worker_sha256"] == source_hash for result in results)
    return artifacts, results


def test_two_rank_large_all_reduce_graph_replays_changed_inputs(tmp_path: Path) -> None:
    _, results = _run_workers(tmp_path, all_reduce_only=True)
    assert all(result["cases"] == [] for result in results)
    assert all(
        result["mixed_all_reduce"]
        == {"rows": [7, 448, 1024, 7], "hidden_size": 5120, "dtype": "bfloat16", "checked_values": [14, 18, 22]}
        for result in results
    )


def test_two_rank_dspark_greedy_graph_consensus(tmp_path: Path) -> None:
    artifacts, results = _run_workers(tmp_path)
    assert all(result["graph_replays"] == 12 for result in results)
    assert [(case["dtype"], case["batch_size"]) for case in results[0]["cases"]] == [
        (dtype, batch) for dtype in ("torch.float32", "torch.bfloat16") for batch in (2, 3)
    ]
    assert all(
        set(digest) == {"confidence", "dense_confidence"}
        for case in results[0]["cases"]
        for digest in case["confidence_sha256"]
    )
    _check_rank_snapshots(artifacts)
    assert results[0]["cases"] == results[1]["cases"]
