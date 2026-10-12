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

"""Register xLLM's native ZMQ publishers with Motor KVConductor.

This configures the KV index subscription; Coordinator engine registration and
request forwarding are separate operations. No KV events are sent over HTTP.
"""

import argparse
import json
import os
import sys
import urllib.request
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.logger import logger


def rank_endpoint(endpoint: str, rank: int) -> str:
    """Use the same base-port + DP-rank convention as the native publisher."""
    address, separator, port = endpoint.rpartition(":")
    if not separator or not endpoint.startswith("tcp://") or not port.isdecimal():
        raise ValueError(f"Expected a TCP endpoint with numeric port: {endpoint}")
    ranked_port = int(port) + rank
    if not 0 < ranked_port <= 65535 or "*" in address:
        raise ValueError("Registration requires a reachable host and valid port")
    return f"{address}:{ranked_port}"


def post_json(base_url: str, route: str, body: dict[str, Any]) -> Any:
    request = urllib.request.Request(
        base_url.rstrip("/") + route,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.load(response)


def register(args: argparse.Namespace) -> list[Any]:
    if args.dp_size <= 0 or args.block_size <= 0:
        raise ValueError("dp-size and block-size must be positive")
    responses = []
    for rank in range(args.dp_size):
        body: dict[str, Any] = {
            "instance_id": args.instance_id,
            "type": "vllm",
            "modelname": args.model,
            "block_size": args.block_size,
            "dp_rank": rank,
            "tenant_id": args.tenant_id,
        }
        if args.unregister:
            responses.append(post_json(args.conductor, "/unregister", body))
            continue
        # A combined NPU+CPU registration keeps each local Host pool attached
        # to this exact DP. A CPU-only Mooncake-style registration fans out by
        # node IP and would incorrectly advertise local copies on other DPs.
        body["store_backend"] = "xllm"
        body["medium_endpoints"] = {
            "npu": rank_endpoint(args.engine_endpoint, rank),
        }
        if args.pool_endpoint:
            body["medium_endpoints"]["cpu"] = rank_endpoint(args.pool_endpoint, rank)
        if args.replay_endpoint:
            body["replay_endpoint"] = rank_endpoint(args.replay_endpoint, rank)
        responses.append(post_json(args.conductor, "/register", body))
    return responses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conductor", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--tenant-id", default="default")
    parser.add_argument("--engine-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--pool-endpoint")
    parser.add_argument("--replay-endpoint")
    parser.add_argument("--unregister", action="store_true")
    args = parser.parse_args()
    responses = register(args)
    logger.info(
        "Motor KV subscription %s for %d DP ranks", "removed" if args.unregister else "registered", args.dp_size
    )
    print(json.dumps(responses))


if __name__ == "__main__":
    main()
