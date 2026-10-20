---
title: "自适应 DSpark"
sidebar:
  order: 85
---

<!-- Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

自适应 DSpark 使用 confidence 反馈和已标定的完整步成本选择验证预算。
参数类型和默认值见 [CLI reference](/zh/cli_reference/#speculativeconfig)。

## 支持的配置

当前自适应 task pipeline 支持 NPU 上的 GLM DSA Python target、带 confidence
head 的 DSpark draft、`num_speculative_tokens=7`、greedy 采样、DP=1/CP=1、
ACL graph、异步调度和 task pipeline。必须启用 graph warmup；这条路径暂不支持
自适应 unified graph。固定步数 DSpark 和其他算法的执行限制保持各自原有约定。

## 启动示例

在已有多 rank 启动脚本的**每个 rank**中加入下列配置。下面展示 16 rank 服务的
rank 0；其余 rank 使用各自的 `node_rank` 和监听端口。启动前创建输出目录，
并选择尚不存在的输出文件：

```bash
mkdir -p /results
export HCCL_OP_EXPANSION_MODE=AIV
xllm --model=/models/target --draft_model=/models/dspark \
  --backend=llm --model_impl=python --npu_kernel_backend=TORCH \
  --nnodes=16 --node_rank=0 --host=127.0.0.1 --port=8000 \
  --master_node_addr=127.0.0.1:27000 \
  --speculative_algorithm=DSpark --num_speculative_tokens=7 \
  --draft_sampling_mode=greedy --enable_adaptive_speculative_decode=true \
  --enable_graph=true --python_graph_backend=aclgraph \
  --disable_graph_warmup=false --enable_schedule_overlap=true \
  --enable_task_pipeline=true --enable_unified_mtp_graph=false \
  --dp_size=1 --cp_size=1 --max_seqs_per_batch=16 \
  --acl_graph_decode_batch_size_limit=16 \
  --adaptive_dspark_cost_table_output=/results/dspark-costs.json
```

客户端显式传入 `temperature=0`。

## 启动标定

标定不需要 HTTP 客户端、手动拟合命令或第二次启动：服务先预热图，再驱动真实
异步 prefill/decode 流水线，在内存中收集完成时间窗口，随后在 CPU 拟合成本，
并在就绪前向所有 worker 安装同一份不可变成本表。正常服务不会增加逐轮计时
Event、耗时采样 D2H 或文件写入；原有结果就绪 Event、结果回读和 confidence
异步反馈仍然保留。

### 内置默认值

未指定标定文件和已保存成本表时，内置默认值为：

| 配置项 | 实际默认值 |
|:-------|:-----------|
| `model` | `additive` |
| `request_buckets` | 实际已捕获的物理 draft request 桶 |
| `target_graph_rows` | 每个 request 桶对应的所有合法已捕获 target 桶，包括完整宽度 |
| `contexts` | 一个区间：prefill 2048 token，KV 覆盖 1–4096 |
| `warmup_steps` | `8` |
| `window_steps` | 每个窗口 `32` 个完整 decode 间隔 |
| `windows_per_point` | `3` |
| `token_seed` | `1` |

默认 KV 上限不会超过模型上下文限制；较短上下文会缩短 prefill，为最坏情况下的
生成进度和两个在途 Slot 预留空间。如果默认采样步数仍然无法容纳，启动会明确
报错，要求提供更小的显式配置。对所有已捕获桶采样可能增加启动时间，可用 JSON
限制标定网格。

### 自定义配置

指定 `--adaptive_dspark_calibration=/configs/dspark.json` 可覆盖默认配置，例如：

```json
{
  "model": "additive",
  "request_buckets": [1, 2, 4, 8, 16],
  "contexts": [
    {"prefill_tokens": 1868, "min_kv_length": 1800, "max_kv_length": 4096}
  ],
  "warmup_steps": 8,
  "window_steps": 32,
  "windows_per_point": 3,
  "token_seed": 1
}
```

### 成本模型与覆盖范围

`model` 支持 `sps` 和 `additive`。一维 SPS 表按物理 target graph 行数 `G`
描述每秒完整步数，成本为 `1000 / SPS(G)` 毫秒；二维加性模型拟合
`bias + alpha(Bcap) + theta(G)` 秒，其中 `Bcap` 是物理 draft request 桶，
不是 HTTP 逻辑并发数。这些都是完整一步的成本，不能再次加上 draft/commit 耗时。
二维网格必须通过共同的 G 值连通；request 桶和可选的 `target_graph_rows` 必须
已经捕获，并保留完整宽度覆盖。上下文区间不能重叠，且必须满足模型、KV 容量和
最坏采样进度的限制。

不管使用默认值还是传入文件，启动日志都会输出
`DSpark adaptive effective configuration`，包含来源、补齐默认项后的生效配置
及实际捕获的图几何。服务过程中不做标定、拟合或 graph recapture。未标定的
KV 区间或 request 桶使用已经预热的完整宽度准入策略，不做 eager 重试。

## 复用已保存成本表

后续兼容的启动可移除标定和输出参数，改为
`--adaptive_dspark_cost_table=/results/dspark-costs.json`。日志会打印成本表来源、
保存的标定配置和生效覆盖范围，并校验权重/量化配置及并行、执行设置的兼容性。
加载表只跳过采样与拟合，不跳过 graph warmup。随机 token 标定和较小训练误差
本身不能证明提速，仍应使用真实请求对比固定步数基线的 TPOT、接受率和接受长度。
