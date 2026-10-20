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

"""Fresh HCCL rank for test_dspark_greedy_consensus.py; no model server.

Production TP collectives use real native HCCL operators. The parent compares
independent Host snapshots; no diagnostic collective uses the TP communicator.
The C++ Host budget planner is intentionally outside this probe's scope.
Both a controlled head and a seeded dense head are tested, not trained weights.
Floating linear weights use ND checkpoint layout; quantized/NZ is not covered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from scripts.logger import logger

_Outputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _make_case(
    rank: int, batch_size: int, dtype: torch.dtype, device: torch.device
) -> tuple[torch.Tensor, Callable[[], _Outputs]]:
    from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
    from xllm.python.models.dspark import DSparkConfidenceHead, DSparkMarkovHead

    hidden_size, vocab_size, markov_rank = 32, 64, 16
    generator = torch.Generator().manual_seed(739)
    # This floating TP/head probe exercises checkpoint ND weights, not the
    # separate packed NZ weight variant that selected legacy aclop MatMul.
    row = RowParallelLinear(hidden_size // 2, hidden_size, 2, dtype=dtype, device=device, use_checkpoint_layout=True)
    head = ColumnParallelLinear(hidden_size, vocab_size // 2, 2, gather_output=True, dtype=dtype, device=device)
    markov = DSparkMarkovHead(vocab_size, vocab_size, markov_rank, dtype, device)
    confidence_head = DSparkConfidenceHead(hidden_size, markov_rank, True, device)
    dense_head = DSparkConfidenceHead(hidden_size, markov_rank, True, device)
    full_weight = torch.randn(hidden_size, hidden_size, generator=generator) / 8
    # This exactly representable channel makes ties/near-ties deliberate, not
    # an accidental consequence of otherwise randomly initialized weights.
    full_weight[0].zero_()
    full_weight[0, 0] = 1
    row.weight.copy_(full_weight[:, rank * 16 : (rank + 1) * 16])
    full_head = torch.randn(vocab_size, hidden_size, generator=generator) / 8
    head.weight.copy_(full_head[rank * 32 : (rank + 1) * 32])
    markov.markov_w1.weight.copy_(torch.randn(vocab_size, markov_rank, generator=generator) / 8)
    markov.markov_w2.weight.copy_(torch.randn(vocab_size, markov_rank, generator=generator) / 8)
    confidence_head.proj.weight.zero_()
    confidence_head.proj.weight[0, 0] = 1
    confidence_head.proj.bias.zero_()
    # The dense FP32 dot product uses both full hidden and Markov components.
    dense_weight = torch.randn(1, hidden_size + markov_rank, generator=generator) / 8
    assert bool((dense_weight != 0).all())
    dense_head.proj.weight.copy_(dense_weight)
    dense_head.proj.bias.fill_(0.125)
    local = torch.empty(batch_size * 7, hidden_size // 2, dtype=dtype, device=device)
    anchors = torch.arange(batch_size, dtype=torch.int64, device=device)
    proposals = torch.empty(batch_size, 7, dtype=torch.int64, device=device)
    confidence = torch.empty(batch_size, 7, dtype=torch.float32, device=device)
    widths = torch.empty(batch_size, dtype=torch.int32, device=device)
    dense_confidence = torch.empty_like(confidence)
    dense_widths = torch.empty_like(widths)
    budget = 2 * batch_size + 1

    def operation() -> _Outputs:
        hidden = row(local)
        logits = head(hidden).view(batch_size, 7, vocab_size)
        previous = anchors
        for column in range(7):
            previous = (logits[:, column] + markov.bias(previous)).argmax(dim=-1)
            proposals[:, column].copy_(previous)
        preceding = torch.cat((anchors.unsqueeze(1), proposals[:, :6]), dim=1)
        hidden_matrix = hidden.view(batch_size, 7, hidden_size)
        markov_embedding = markov.embed(preceding)
        confidence.copy_(confidence_head(hidden_matrix, markov_embedding))
        dense_confidence.copy_(dense_head(hidden_matrix, markov_embedding))
        torch.ops.xllm_ops.dspark_allocate_verify_prefixes(confidence, widths, budget)
        torch.ops.xllm_ops.dspark_allocate_verify_prefixes(dense_confidence, dense_widths, budget)
        return hidden, logits, proposals, confidence, widths, dense_confidence, dense_widths

    return local, operation


def _bits(value: torch.Tensor) -> torch.Tensor:
    import torch_npu

    # The oracle compares logical bits, not any padded physical NZ storage.
    value = torch_npu.npu_format_cast(value, torch_npu.Format.ND).contiguous()
    if value.dtype in (torch.float16, torch.bfloat16):
        return value.view(torch.int16).to(torch.int32).flatten()
    return value.view(torch.int32).flatten()


def _check_local(outputs: _Outputs, batch_size: int, label: str, artifacts: Path) -> dict[str, str]:
    names = ("hidden", "gathered_logits", "proposals", "confidence", "widths", "dense_confidence", "dense_widths")
    rank = dist.get_rank()
    local_bits = [_bits(output).cpu().clone() for output in outputs]
    # The parent performs strict cross-rank comparison after both ranks exit.
    # No diagnostic HCCL collective can disturb the captured TP communicator.
    local_reports = [
        {
            "name": name,
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "int32_bits": bits.tolist(),
        }
        for name, output, bits in zip(names, outputs, local_bits, strict=True)
    ]
    local_path = artifacts / f"rank-{rank}-b{batch_size}-{outputs[0].dtype}-{label}-local.json"
    local_payload = {"rank": rank, "batch_size": batch_size, "label": label, "outputs": local_reports}
    local_path.write_text(json.dumps(local_payload, indent=2) + "\n", encoding="utf-8")
    digests = {
        name: hashlib.sha256(bits.numpy().tobytes()).hexdigest()
        for name, bits in zip(names, local_bits, strict=True)
        if name.endswith("confidence")
    }
    for index in (3, 5):
        confidence, widths = outputs[index].cpu(), outputs[index + 1].cpu()
        assert bool(torch.isfinite(confidence).all())
        assert bool(((confidence >= 0) & (confidence <= 1)).all())
        assert bool(((widths >= 1) & (widths <= 8)).all())
        assert int(widths.sum()) == batch_size + 2 * batch_size + 1
    return digests


def _run_case(
    rank: int, batch_size: int, dtype: torch.dtype, device: torch.device, artifacts: Path
) -> dict[str, object]:
    local, operation = _make_case(rank, batch_size, dtype, device)
    reference_input, reference = _make_case(rank, batch_size, dtype, device)
    generator = torch.Generator().manual_seed(983)
    normal = torch.randn(batch_size * 7, 32, generator=generator)
    ties = torch.zeros_like(normal)
    near_ties = ties.clone()
    near_ties[:, 0] = torch.arange(batch_size * 7) * 2**-18
    scenarios = (("normal", normal), ("ties", ties), ("near_ties", near_ties), ("changed", normal + 0.75))

    def update(full_input: torch.Tensor) -> None:
        shard = full_input[:, rank * 16 : (rank + 1) * 16]
        local.copy_(shard)
        reference_input.copy_(shard)

    update(normal)
    operation()
    warm = operation()
    torch.npu.synchronize()
    digests = [_check_local(warm, batch_size, "eager", artifacts)]
    stream = torch.npu.Stream(device=device)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        captured = operation()
    torch.npu.synchronize()
    pointers = [value.data_ptr() for value in captured]
    for label, full_input in scenarios[1:]:
        update(full_input)
        for value in captured:
            value.fill_(float("nan") if value.is_floating_point() else -99)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        digests.append(_check_local(captured, batch_size, label, artifacts))
        expected = reference()
        torch.npu.synchronize()
        for actual, oracle in zip(captured, expected, strict=True):
            assert torch.equal(_bits(actual).cpu(), _bits(oracle).cpu()), f"{label}: replay differs from eager"
        if label == "ties":
            budget = 2 * batch_size + 1
            expected_widths = [1 + budget // batch_size + (index < budget % batch_size) for index in range(batch_size)]
            assert captured[4].cpu().tolist() == expected_widths
        if label == "near_ties":
            confidence = captured[3].cpu()
            assert 0 < float(confidence.max() - confidence.min()) < 1e-4
        assert [value.data_ptr() for value in captured] == pointers
    for name in ("confidence", "dense_confidence"):
        assert digests[0][name] != digests[-1][name], f"changed graph inputs did not change {name}"
    return {"batch_size": batch_size, "dtype": str(dtype), "confidence_sha256": digests}


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("device must be nonnegative")
    import torch_npu  # noqa: F401

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    torch.set_num_threads(1)
    # Standalone fresh ranks have no tests/python/conftest bootstrap. Use the
    # public imports, not a custom extension loader or a sys.path modification.
    from xllm import xllm_export  # noqa: F401
    from xllm.python import initialize_runtime

    initialize_runtime()
    from xllm.python.distributed import collectives
    from xllm.python.distributed.npu.hccl import get_hccl_comm

    dist.init_process_group(
        backend="hccl",
        init_method=(args.artifact_dir.resolve() / "rendezvous").as_uri(),
        rank=args.rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    # Register the real HCCL world as this probe's TP cohort, matching the
    # existing collective test fixture; no fake group/operation is installed.
    group = dist.distributed_c10d._get_default_group()
    collectives._groups[("tp", str(device))] = group
    collectives._group_ranks[("tp", str(device))] = (0, 1)
    get_hccl_comm(group, device, initialize=True)
    with torch.no_grad():
        cases = [
            _run_case(args.rank, batch, dtype, device, args.artifact_dir)
            for dtype in (torch.float32, torch.bfloat16)
            for batch in (2, 3)
        ]
    collectives._groups.clear()
    collectives._group_ranks.clear()
    dist.destroy_process_group()
    result = {
        "rank": args.rank,
        "graph_replays": len(cases) * 3,
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases,
    }
    (args.artifact_dir / f"rank-{args.rank}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "DSpark TP rank %d completed %d local graph replay checks; parent verifies TP snapshots",
        args.rank,
        result["graph_replays"],
    )


if __name__ == "__main__":
    _main()
