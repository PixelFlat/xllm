---
title: "编译与运行"
---
<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

## 准备依赖

使用 Linux 开发环境，准备 CMake 3.26 或更新版本、支持 C++20 的编译器、Rust/Cargo、安装了 PyTorch 的 Python、Git 及依赖所需的原生构建工具。service 大部分目标使用 C++17，DeepSeek V4 chat template 目标需要 C++20。Ascend 引擎环境参考[快速开始](/zh/getting_started/quick_start/)准备兼容的 xLLM/CANN 环境。etcd 服务端需从[官方发行版](https://github.com/etcd-io/etcd/releases)单独安装。

service 通过 vcpkg 管理 Boost、gRPC、Protobuf、OpenSSL，以及 KV events 版本中的 ZeroMQ。ZeroMQ 使用 service 自身的 manifest，不依赖 Motor 构建产物。CMake 会检测 PyTorch 的 C++ ABI，构建时需要保持环境一致。`prepare.sh` 为 cpprestsdk 添加补丁，以满足 etcd C++ 客户端对静态库的要求。

## 编译 service

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

复用 vcpkg 时若缺少 Boost locale/interprocess 头文件，使用与架构对应的 triplet（`arm64-linux` 或 `x64-linux`）修复相关包，再执行 CMake。不要复用不同架构或 ABI 的 vcpkg 构建产物。

## 启动 etcd

在独立终端运行。下列回环地址仅用于单机示例；多机部署需改成所有参与进程均可访问的地址。

```bash
/path/to/etcd --name=default --data-dir=/path/to/etcd-data \
  --listen-client-urls=http://127.0.0.1:2379 \
  --advertise-client-urls=http://127.0.0.1:2379 \
  --listen-peer-urls=http://127.0.0.1:2380 \
  --initial-advertise-peer-urls=http://127.0.0.1:2380 \
  --initial-cluster=default=http://127.0.0.1:2380
```

## 启动 service

在 service 仓库运行 HTTP/RPC 合一入口：

```bash
./build/xllm_service/xllm_master_serving \
  --etcd_addr=127.0.0.1:2379 \
  --etcd_namespace=demo \
  --http_server_port=8888 \
  --rpc_server_port=8889 \
  --load_balance_policy=RR \
  --tokenizer_path=/models/Qwen2-7B-Instruct
```

`tokenizer_path` 应包含与引擎模型一致的 tokenizer 配置。HTTP 接收客户端请求，RPC 与引擎交互。完整实例组注册前，HTTP 端口可能尚未监听。

## 启动引擎

选择以下一种部署方式，并保持 etcd 地址和 namespace 与 service 一致。示例每个实例使用一张 Ascend 卡；请保留当前引擎版本和模型所需的硬件、并行与推理参数，参见[启动 xLLM](/zh/getting_started/launch_xllm/)。

### 非 PD 部署

```bash
ASCEND_RT_VISIBLE_DEVICES=0 /path/to/xllm --model=/models/Qwen2-7B-Instruct \
  --port=8000 --master_node_addr=127.0.0.1:19888 \
  --nnodes=1 --node_rank=0 \
  --enable_service_routing=true --instance_role=DEFAULT \
  --etcd_addr=127.0.0.1:2379 --etcd_namespace=demo
```

需要实例池时，在其他设备启动更多 DEFAULT 实例，为它们分别配置不同的 HTTP 和 master-node 端口。

### PD 部署

本示例与上述 DEFAULT 实例池分开部署。先按 [PD 分离部署](/zh/features/disagg_pd/)准备设备网络与容器映射；Ascend 环境包括对 `/etc/hccn.conf` 的访问。

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

传输端口属于引擎配置，与 service 的 HTTP/RPC 端口不同。Prefix cache、chunked prefill、图模式、投机解码及并行能力由引擎与模型决定，某次测试中的参数不是 service 的强制限制。

## 发送请求

实例组就绪后，请求 service 的 HTTP 端口：

```bash
curl -sS http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen2-7B-Instruct","messages":[{"role":"user","content":"Explain prefix caching."}],"max_tokens":64,"stream":false}'
```

`model` 填写实际服务的模型名称。

## 常用 service 参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `server_host` | 空 | HTTP 与 RPC 共用的监听地址 |
| `http_server_port` / `rpc_server_port` | 8888 / 8889 | 客户端入口与引擎交互端口 |
| `http_server_num_threads` / `rpc_server_num_threads` | 32 / 32 | 监听线程数 |
| `http_server_max_concurrency` / `rpc_server_max_concurrency` | 128 / 128 | 请求并发上限 |
| `http_server_idle_timeout_s` / `rpc_server_idle_timeout_s` | -1 / -1 | 空闲连接超时；-1 表示不启用 |
| `etcd_namespace` | 空 | 隔离注册与协调信息 |
| `load_balance_policy` | RR | 调度策略 |

缓存感知路由的配置参见 [KV events 与 CAR](/zh/xllm_service/kv_events/)。
