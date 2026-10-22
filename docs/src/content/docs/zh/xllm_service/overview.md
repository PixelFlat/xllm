---
title: "概述"
---
<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

[xLLM Service](https://github.com/xLLM-AI/xllm-service) 是 xLLM 的集群服务层，负责接收推理请求、发现已注册的引擎、跟踪实例健康与负载，并调度请求。每个 service 集群部署同一个模型，并保持引擎配置兼容。

## 组件与请求链路

| 组件 | 职责 |
| --- | --- |
| HTTP 服务 | 接收客户端请求，包括 `/v1/chat/completions` 和 `/v1/completions` |
| RPC 服务 | 与 xLLM 实例交互并接收状态更新 |
| etcd | 保存实例注册与协调元数据，支持实例发现和负载同步 |
| 实例管理器 | 维护成员、健康状态及可调度候选实例 |
| 调度器 | 通过 RR、SLO_AWARE、CAR 等策略选择实例 |
| xLLM 引擎 | 加载权重、执行推理并管理实际 KV tensor |

依次启动 etcd、service 和引擎。引擎由用户手动或通过脚本启动，并注册到相同的 etcd 地址与 namespace。service 自动发现注册信息；本流程不支持通过 JSON/YAML 部署文件自动拉起引擎。

非分离部署由 DEFAULT 实例完成整个请求。Prefill/Decode（PD）分离部署由调度器选择 P、D 实例，P 执行 prefill 并通过已配置的引擎链路传输 KV，D 向 service 返回生成结果。

只有存在合适的可用实例组时，service 才启动 HTTP 监听；实例组全部不可用时停止监听，恢复后重新启动。

## KV 感知与能力边界

可选的 [KV events 目录与 CAR 路由](/zh/xllm_service/kv_events/) 在后台异步维护 Device/Host 前缀元数据，调度器在请求路径查询本地目录。目录不存储或迁移 KV tensor；缓存分配、卸载与实际复用由引擎负责。

早期架构中的 Global KV Cache Manager、动态 KV 迁移、Event Plane、Planner 包含设计方向，不能据此认定当前已经实现自动迁移或扩缩容规划器。本文流程不包含自动托管引擎、自动扩缩容或 Kubernetes controller。KV 目录所需的 service 版本前提集中说明在 KV events 页面。

下一步：[编译与运行](/zh/xllm_service/getting_started/)。
