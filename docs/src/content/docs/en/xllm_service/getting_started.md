---
title: "Build and run"
---
<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

## Prepare dependencies

Build service in a Linux development environment with CMake 3.26 or later, a compiler supporting C++20, Rust/Cargo, Python with PyTorch, Git and the native build tools required by its dependencies. Most service targets use C++17; the DeepSeek V4 chat-template target requires C++20. For Ascend engines, prepare the compatible xLLM/CANN environment using [Quick start](/en/getting_started/quick_start/). Install the etcd server separately using the [official releases](https://github.com/etcd-io/etcd/releases).

Service uses vcpkg for dependencies including Boost, gRPC, Protobuf, OpenSSL and ZeroMQ. CMake detects PyTorch's C++ ABI; use a consistent build environment. `prepare.sh` patches cpprestsdk for the etcd C++ client's static-library requirement.

## Build service

```bash
git clone --recurse-submodules https://github.com/xLLM-AI/xllm-service.git
cd xllm-service
bash prepare.sh
mkdir -p build
cd build
cmake ..
make -j8
cd ..
```

If Boost locale/interprocess headers are missing when reusing a vcpkg installation, repair those packages with the triplet matching your architecture (`arm64-linux` or `x64-linux`) and rerun CMake. Avoid copying a vcpkg build from a different architecture or ABI.

## Start etcd

Run the following in a separate terminal. The loopback addresses below describe a single-host example; for multiple hosts, use addresses reachable by every participant.

```bash
/path/to/etcd --name=default --data-dir=/path/to/etcd-data \
  --listen-client-urls=http://127.0.0.1:2379 \
  --advertise-client-urls=http://127.0.0.1:2379 \
  --listen-peer-urls=http://127.0.0.1:2380 \
  --initial-advertise-peer-urls=http://127.0.0.1:2380 \
  --initial-cluster=default=http://127.0.0.1:2380
```

## Start service

From the service repository, start the combined HTTP/RPC entry point:

```bash
./build/xllm_service/xllm_master_serving \
  --etcd_addr=127.0.0.1:2379 \
  --etcd_namespace=demo \
  --http_server_port=8888 \
  --rpc_server_port=8889 \
  --load_balance_policy=RR \
  --tokenizer_path=/models/Qwen2-7B-Instruct
```

`tokenizer_path` must contain the tokenizer configuration for the engines' model. HTTP accepts client requests; RPC communicates with engines. The HTTP port may remain closed until a complete instance group registers. Engines must be able to reach the RPC port; in PD mode, Decode returns all generated tokens, including the first, through this port.

## Start engines

Choose one of the following deployments. Keep the etcd address and namespace identical to service. These commands illustrate one Ascend device per instance; retain the model, hardware and parallelism options required by your engine build, as described in [Launch xLLM](/en/getting_started/launch_xllm/).

### Non-PD deployment

```bash
ASCEND_RT_VISIBLE_DEVICES=0 /path/to/xllm --model=/models/Qwen2-7B-Instruct \
  --port=8000 --master_node_addr=127.0.0.1:19888 \
  --nnodes=1 --node_rank=0 \
  --enable_service_routing=true --instance_role=DEFAULT \
  --etcd_addr=127.0.0.1:2379 --etcd_namespace=demo
```

Start further DEFAULT instances on different devices with unique HTTP and master-node ports to form a pool.

### PD deployment

Use a separate deployment from the DEFAULT example. Prepare the device networking and container mounts described in [Disaggregated PD](/en/features/disagg_pd/), including access to `/etc/hccn.conf` on Ascend.

```bash
# Decode: start first
ASCEND_RT_VISIBLE_DEVICES=1 /path/to/xllm --model=/models/Qwen2-7B-Instruct \
  --port=8020 --master_node_addr=127.0.0.1:19898 \
  --nnodes=1 --node_rank=0 --enable_service_routing=true \
  --enable_disagg_pd=true --instance_role=DECODE \
  --etcd_addr=127.0.0.1:2379 --etcd_namespace=demo \
  --transfer_listen_port=26100 --disagg_pd_port=7787

# Prefill: run in another terminal after Decode is ready
ASCEND_RT_VISIBLE_DEVICES=0 /path/to/xllm --model=/models/Qwen2-7B-Instruct \
  --port=8010 --master_node_addr=127.0.0.1:19888 \
  --nnodes=1 --node_rank=0 --enable_service_routing=true \
  --enable_disagg_pd=true --instance_role=PREFILL \
  --etcd_addr=127.0.0.1:2379 --etcd_namespace=demo \
  --transfer_listen_port=26000 --disagg_pd_port=7777
```

The transfer ports are engine settings, separate from service's HTTP/RPC ports. Prefix cache, chunked prefill, graph mode, speculative decoding and parallelism support are determined by the engine and model; service does not impose the settings of an individual benchmark.

## Send a request

After the engine group is ready, send requests to service's HTTP port:

```bash
curl -sS http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2-7B-Instruct","messages":[{"role":"user","content":"Explain prefix caching."}],"max_tokens":64,"stream":false}'
```

Use the served model name.

## Useful service flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `server_host` | empty | Bind address shared by HTTP and RPC listeners |
| `http_server_port` / `rpc_server_port` | 8888 / 8889 | Client and engine-facing ports |
| `http_server_num_threads` / `rpc_server_num_threads` | 32 / 32 | Listener thread counts |
| `http_server_max_concurrency` / `rpc_server_max_concurrency` | 128 / 128 | Request concurrency limits |
| `http_server_idle_timeout_s` / `rpc_server_idle_timeout_s` | -1 / -1 | Idle connection timeout; -1 disables it |
| `etcd_namespace` | empty | Isolates registration and coordination keys |
| `load_balance_policy` | RR | Scheduling policy |

For optional cache-aware routing, continue with [KV events and CAR](/en/xllm_service/kv_events/).
