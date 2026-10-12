# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://github.com/xLLM-AI/xllm/blob/main/LICENSE
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the production prefix-cache fixture against an unmodified KVConductor.

Usage: python tests/core/kv_cache/prefix/test_motor_kv_events.py
    --fixture /path/to/kv_cache_event_fixture --conductor /path/to/kv-conductor
Requires pyzmq and msgpack. Does not allocate NPU tensors or load a model.
"""

import argparse
import json
import select
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, TextIO

import msgpack
import zmq

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.logger import logger
from tools.register_motor_kv_events import post_json, register


def free_ports(count: int, excluded: set[int]) -> int:
    for _ in range(100):
        sockets = []
        try:
            first = socket.socket()
            sockets.append(first)
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            if port + count > 65535 or any(p in excluded for p in range(port, port + count)):
                continue
            for offset in range(1, count):
                extra = socket.socket()
                sockets.append(extra)
                extra.bind(("127.0.0.1", port + offset))
            excluded.update(range(port, port + count))
            return port
        except OSError:
            continue
        finally:
            for bound in sockets:
                bound.close()
    raise RuntimeError("Could not find free adjacent ports")


def read_line(process: subprocess.Popen[str]) -> str:
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], 10)
    if not ready:
        raise TimeoutError("Fixture did not respond within 10 seconds")
    line = process.stdout.readline().strip()
    if not line:
        raise RuntimeError(f"Fixture exited: {process.poll()}")
    return line


def command(process: subprocess.Popen[str], **body: Any) -> str:
    assert process.stdin is not None
    process.stdin.write(json.dumps(body) + "\n")
    process.stdin.flush()
    return read_line(process)


def start_conductor(binary: str, port: int, log: TextIO) -> subprocess.Popen[str]:
    process = subprocess.Popen([binary, "--host", "127.0.0.1", "--port", str(port)], stdout=log, stderr=log, text=True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"KVConductor exited: {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return process
        except (OSError, urllib.error.URLError):
            time.sleep(0.05)
    process.terminate()
    process.wait(timeout=5)
    raise TimeoutError("KVConductor did not become ready")


def terminate(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def run(fixture_binary: str, conductor_binary: str, report_path: Path) -> None:
    ports: set[int] = set()
    motor_port = free_ports(1, ports)
    engine_port = free_ports(2, ports)
    pool_port = free_ports(2, ports)
    replay_port = free_ports(2, ports)
    engine = f"tcp://127.0.0.1:{engine_port}"
    pool = f"tcp://127.0.0.1:{pool_port}"
    replay = f"tcp://127.0.0.1:{replay_port}"
    url = f"http://127.0.0.1:{motor_port}"
    registration = argparse.Namespace(
        conductor=url,
        instance_id="xllm-fixture",
        model="fixture",
        block_size=2,
        dp_size=2,
        tenant_id="default",
        engine_endpoint=engine,
        pool_endpoint=pool,
        replay_endpoint=replay,
        unregister=False,
    )
    observed: dict[str, Any] = {}
    tokens = [10, 11, 12, 13, 14, 15]
    other = [20, 21, 22, 23, 24, 25]
    fixture: subprocess.Popen[str] | None = None
    conductor: subprocess.Popen[str] | None = None
    with tempfile.TemporaryDirectory(prefix="xllm-motor-test-") as directory:
        fixture_log_path = Path(directory) / "fixture.log"
        motor_log_path = Path(directory) / "motor.log"
        with fixture_log_path.open("w+") as fixture_log, motor_log_path.open("w+") as motor_log:
            context = zmq.Context()
            context.setsockopt(zmq.LINGER, 0)
            try:
                conductor = start_conductor(conductor_binary, motor_port, motor_log)
                config = {
                    "publisher": "zmq",
                    "endpoint": engine,
                    "motor_pool_endpoint": pool,
                    "replay_endpoint": replay,
                    "snapshot_interval_seconds": 2,
                }
                fixture = subprocess.Popen(
                    [fixture_binary, json.dumps(config)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=fixture_log,
                    text=True,
                    bufsize=1,
                )
                assert read_line(fixture) == "ready"
                # Populate caches before subscription to exercise snapshot recovery.
                assert command(fixture, op="store", rank=0, tokens=tokens) == "ok"
                assert command(fixture, op="store", rank=1, tokens=other) == "ok"
                register(registration)

                def expect(name: str, query_tokens: list[int], rank: int, npu: int, cpu: int) -> None:
                    deadline = time.monotonic() + 8
                    result: dict[str, Any] = {}
                    while time.monotonic() < deadline:
                        result = post_json(
                            url, "/query", {"model": "fixture", "block_size": 2, "token_ids": query_tokens}
                        )
                        counts = result.get("default", {}).get("xllm-fixture", {}).get("DP", {}).get(str(rank), {})
                        if counts.get("npu_blocks", 0) == npu and counts.get("cpu_blocks", 0) == cpu:
                            observed[name] = result
                            logger.info("%s passed: DP %d, NPU %d, CPU %d", name, rank, npu, cpu)
                            return
                        time.sleep(0.05)
                    raise AssertionError(f"{name}: expected NPU={npu}, CPU={cpu}, got {result}")

                expect("device", tokens, 0, 3, 0)
                expect("dp_isolation", tokens, 1, 0, 0)
                expect("other_dp", other, 1, 3, 0)

                # Inspect the real multipart wire and replay termination marker.
                with context.socket(zmq.SUB) as subscriber:
                    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
                    subscriber.setsockopt(zmq.RCVTIMEO, 3000)
                    subscriber.connect(engine)
                    frames = subscriber.recv_multipart()
                    assert len(frames) == 3 and len(frames[1]) == 8
                    batch = msgpack.unpackb(frames[2], raw=False)
                    assert batch[2] == 0
                    stored = next(event for event in batch[1] if event["type"] == "BlockStored")
                    assert stored["token_ids"] == tokens[:2] and stored["block_size"] == 2
                with context.socket(zmq.DEALER) as dealer:
                    dealer.setsockopt(zmq.RCVTIMEO, 3000)
                    dealer.connect(replay)
                    dealer.send_multipart([b"", (0).to_bytes(8, "big")])
                    replay_frames = dealer.recv_multipart()
                    assert len(replay_frames) == 3
                    assert any(
                        event["type"] == "BlockStored" for event in msgpack.unpackb(replay_frames[2], raw=False)[1]
                    )
                    assert int.from_bytes(dealer.recv_multipart()[1], "big") == 2**64 - 1
                observed["wire_and_replay"] = "passed"

                assert command(fixture, op="copy", rank=0, tokens=tokens) == "ok"
                state = json.loads(command(fixture, op="state"))
                assert sum(event["medium"] == "cpu" for event in state["events"]) == 3
                expect("replicas_counted_once", tokens, 0, 3, 0)
                assert command(fixture, op="clear", rank=0, medium="gpu") == "ok"
                expect("host_after_device_eviction", tokens, 0, 0, 3)
                expect("host_dp_isolation", tokens, 1, 0, 0)
                assert command(fixture, op="store", rank=0, tokens=tokens[:2]) == "ok"
                expect("mixed_tiers", tokens, 0, 1, 2)
                assert command(fixture, op="clear", rank=0, medium="gpu") == "ok"
                expect("host_before_restart", tokens, 0, 0, 3)

                terminate(conductor)
                conductor = start_conductor(conductor_binary, motor_port, motor_log)
                register(registration)
                expect("conductor_restart", tokens, 0, 0, 3)
                expect("other_dp_after_restart", other, 1, 3, 0)
                assert command(fixture, op="clear", rank=0, medium="cpu") == "ok"
                expect("host_eviction", tokens, 0, 0, 0)
                expect("unaffected_dp", other, 1, 3, 0)
                registration.unregister = True
                register(registration)
                expect("unregister", other, 1, 0, 0)
                assert command(fixture, op="state")
                report_path.write_text(json.dumps(observed, indent=2) + "\n")
                logger.info("Motor integration passed; report: %s", report_path)
            except Exception:
                fixture_log.flush()
                motor_log.flush()
                logger.error("Fixture log:\n%s", fixture_log_path.read_text()[-8000:])
                logger.error("Motor log:\n%s", motor_log_path.read_text()[-12000:])
                raise
            finally:
                terminate(fixture)
                terminate(conductor)
                context.term()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--conductor", required=True)
    parser.add_argument("--report", type=Path, default=Path("motor_kv_events_report.json"))
    args = parser.parse_args()
    run(args.fixture, args.conductor, args.report)


if __name__ == "__main__":
    main()
