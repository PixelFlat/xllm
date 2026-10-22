---
title: "Overview"
---
<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

[xLLM Service](https://github.com/xLLM-AI/xllm-service) is the cluster service layer for xLLM. It accepts inference requests, discovers registered engines, tracks their health and load, and schedules requests to instances. Deploy a single model with compatible engine settings in each service cluster.

## Components and request flow

| Component | Responsibility |
| --- | --- |
| HTTP service | Receives client requests, including `/v1/chat/completions` and `/v1/completions` |
| RPC service | Communicates with xLLM instances and receives status updates |
| etcd | Stores instance registration and coordination metadata; supports discovery and load synchronization |
| Instance manager | Maintains instance membership, health state and schedulable candidates |
| Scheduler | Selects instances using policies such as RR, SLO_AWARE and CAR |
| xLLM engines | Load model weights, execute inference and manage actual KV tensors |

Start etcd, then service, then the engines. Engines are started manually or by your own scripts and register with the same etcd address and namespace. Service discovers these registrations; it does not launch engines from a JSON/YAML deployment file.

In a non-disaggregated deployment, a DEFAULT engine executes the complete request. In a prefill/decode (PD) deployment, the scheduler selects a Prefill (P) and Decode (D) instance, P performs prefill and transfers KV using the configured engine transport, and D returns generated output to service.

The service HTTP listener becomes available only when a suitable instance group is available. It is stopped when no suitable group remains and starts again when instances recover.

## KV awareness and scope

The optional [KV events directory and CAR routing](/en/xllm_service/kv_events/) asynchronously track Device and Host prefix metadata. The scheduler queries this local directory during routing. The directory neither stores nor migrates KV tensors; the engines retain ownership of cache allocation, offload and reuse.

Automatic engine provisioning, autoscaling, a Kubernetes controller and cross-instance KV migration are outside the deployment workflow described here.

Continue with [Build and run](/en/xllm_service/getting_started/).
