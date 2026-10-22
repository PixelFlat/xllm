---
title: "KV events and CAR"
---
<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

## Version requirements

This page describes the tiered KV events directory and CAR implementation introduced by [xllm-service PR #93](https://github.com/xLLM-AI/xllm-service/pull/93) (reference revision `eb388de`). Build a service revision containing that change and use an xLLM engine supporting the event and full-snapshot protocols described here. Older service releases require an upgrade. See [Build and run](/en/xllm_service/getting_started/) for the base deployment.

## What the directory tracks

Service subscribes to xLLM ZeroMQ KV events in background threads and maintains a prefix directory separated by instance, DP rank and Device/Host tier. It stores metadata, never KV tensors. `kv_directory` handles subscriptions and discovery, `kv_index` maintains tiered prefixes, and `kv_event_decoder` decodes events.

CAR and the internal query endpoint share this directory. This revision replaces the old GlobalKVCacheMgr, KV heartbeat consumption and KV etcd reads/writes/watch. Instance discovery, health checks, load heartbeats and load synchronization through etcd remain. RR and SLO_AWARE retain their selection logic.

## Configure engines and service

Retain the engine's model and inference settings and add:

```bash
--port=8000 \
--enable_prefix_cache=true \
--block_size=128 \
--kv_events_config='{"publisher":"zmq","endpoint":"tcp://*:18000","replay_endpoint":"tcp://*:38000","topic":"xllm-kv"}'
```

Host events require local Host offload to be enabled in the engine and the copy to have completed successfully. In PD mode, subscribe to P: reserved blocks on D are not reusable prefill KV. Existing engine restrictions on flat-text prefix caching still apply.

Engines must enable service routing and register with the same etcd address and namespace. Start them manually or through your own scripts or infrastructure. Discovery and subscription do not imply that service launches engines.

Save the following as `/path/to/kv_events_discovery_config.json`:

```json
{
  "discovery": {
    "enabled": true,
    "block_size": 128,
    "topic": "xllm-kv",
    "event_port_offset": 10000,
    "replay_port_offset": 30000,
    "refresh_interval_ms": 1000
  }
}
```

Add these flags to the service command from [Build and run](/en/xllm_service/getting_started/):

```bash
--load_balance_policy=CAR \
--car_prefill_lambda=0.5 \
--car_host_hit_weight=0.5 \
--kv_events_config_file=/path/to/kv_events_discovery_config.json
```

Service's `kv_events_config_file` takes a JSON **file path**; the engine's `kv_events_config` takes a JSON **string**. An empty file path disables the directory; CAR logs a warning and uses zero prefill hit ratios. This integration is in `xllm_master_serving`, not the standalone example HTTP/RPC programs. Configuration is validated at startup; restart service after editing it.

## Automatic discovery

A background task reads the local instance snapshot already maintained by `InstanceMgr` through etcd watch. It does not scan etcd again or connect to publishers in request handlers. For a registered engine HTTP address `host:port`:

```text
event endpoint = tcp://host:(port + event_port_offset + dp_rank)
recovery endpoint = tcp://host:(port + replay_port_offset + dp_rank)
```

For `127.0.0.1:8000` with two DP ranks, event ports are 18000/18001 and recovery ports are 38000/38001. The base is the engine's business HTTP port, not its brpc or KV-transfer port. Engines must publish on matching endpoints: service cannot change a running engine's parameters. Use the same model and compatible KV settings throughout the cluster; no additional model grouping is configured. The discovery `block_size` must match published events. The older service `--block_size` flag does not configure this index.

Discovery subscribes to xLLM DEFAULT/PREFILL instances and MIX instances currently acting as PREFILL. DECODE and other backends are excluded. A lease-loss grace period retains subscriptions; entering SUSPECT or leaving the instance directory removes their subscriptions and query entries. Refresh defaults to 1000 ms and accepts 50–60000 ms. Health detection and grace-period timing follow the existing instance state machine.

A registration refresh with the same `incarnation_id` preserves the index. Identity, address or DP topology changes update subscriptions. A new process reusing an address receives no cache credit before its snapshot arrives. For older registrations, `register_ts_ms` helps identify replacements; when both identity fields are absent, reconnect and snapshot recovery provide the remaining protection.

Port overflow or overlapping endpoints exclude the affected instances and log errors. The default `max_streams` is 128 (allowed range 1–128). When capacity is exhausted, whole instances are skipped in name order instead of retaining only some of their DP ranks.

## Resource and recovery limits

These optional top-level fields belong beside `discovery`. They limit metadata and recovery resources, not engine KV capacity.

| Field | Default | Purpose |
| --- | --- | --- |
| `max_streams` | 128 | Maximum subscription streams; one per instance/DP |
| `max_blocks_per_stream` | 262144 | Block metadata limit, including parent-chain and tokenless-event mappings |
| `max_message_bytes` | 8388608 (8 MiB) | Maximum live message or snapshot reply size |
| `max_buffered_bytes` | 16777216 (16 MiB) | Live-delta buffering budget during recovery, including wire bytes and decoded records |
| `replay_timeout_ms` | 2000 | Timeout for a complete snapshot; an incomplete stream remains unsynchronized and retries |
| `resync_interval_ms` | 30000 | Periodic snapshot reconciliation interval |

### Static streams

Explicit `streams` lists are available for protocol tests and observation. Production CAR uses discovery with matching process identity. `discovery` and a nonempty `streams` list are mutually exclusive. Each stream describes one instance/DP:

| Field | Meaning |
| --- | --- |
| `instance_name` | Instance identity, preferably matching the instance directory |
| `dp_rank` | DP rank; default 0, must match the event batch |
| `block_size` | Tokens per block; default 128, must match Stored events |
| `endpoint` | Required connectable TCP event endpoint; do not use `*` or `0.0.0.0` |
| `replay_endpoint` | Required connectable TCP snapshot endpoint |
| `topic` | Exact publisher topic; default empty string |
| `hash_seed` | Internal directory content-key seed; default 1024, independent of engine hashing |

Configure every DP endpoint separately, matching the engine's base-port-plus-rank rule. Event and recovery ranges must not overlap. Duplicate instance/DP pairs or endpoints are rejected. Each stream owns a background thread and index; connection failures leave it unsynchronized while it retries. Static streams without consistent process identity do not provide CAR cache credit.

## Query the directory

Once service is ready, use its existing HTTP port:

```bash
curl -sS http://127.0.0.1:8888/v1/internal/kv-cache/query \
  -H 'Content-Type: application/json' \
  -d '{"token_ids":[1,2,3,4]}'
```

If `internal_api_token` is configured, also supply the `X-Internal-Token` header. This endpoint follows normal HTTP readiness and does not open the port early. Input must already be tokenized with the correct chat template; the endpoint does not tokenize text. Empty `token_ids` return stream synchronization and residency information.

The `matches` array separates each instance/DP:

| Field | Meaning |
| --- | --- |
| `synchronized` | Valid snapshot plus continuous deltas; false means zero hits |
| `sequence` | Last indexed event sequence |
| `device_hit_tokens` | Reusable continuous-prefix tokens resident on Device |
| `host_only_hit_tokens` | Prefix tokens available only on Host |
| `reusable_prefix_tokens` | Sum of the two hit fields, without counting tier replicas twice |
| `device_blocks` / `host_blocks` | Resident block counts per tier; replicas can overlap |
| `error` | Reason the stream is unsynchronized |

Only complete blocks count. Matching stops at the first block absent from both tiers on that instance/DP. Prefixes are not assembled across DP ranks; Host cache is not shared between instances merely because they run on the same machine. Results reflect an asynchronous observation; actual reuse depends on engine residency when the request executes.

## CAR scoring

Routing obtains schedulable P/DEFAULT and D candidates and their current load snapshots, then synchronously queries the in-process directory. The request path performs no ZMQ, HTTP or etcd directory lookup. Instances with no initial load report, invalid load or SUSPECT state are excluded. The primary service can use fresh heartbeat load without waiting for periodic etcd synchronization. MIX candidates follow their current role; only DEFAULT can serve alone when there is no D.

P/DEFAULT selects the **smallest** cost:

```text
kv_hit_ratio = clamp((Device hit tokens + host_weight * Host-only hit tokens)
                     / input tokens, 0, 1)
queue_cost = waiting_requests_num / maximum waiting_requests_num among P candidates
score = lambda * (1 - kv_hit_ratio) + (1 - lambda) * queue_cost
```

`--car_prefill_lambda` defaults to 0.5 and accepts [0, 1]. Larger values favor cache reuse; 0 compares only queues and 1 compares only cache-miss cost. Queue cost is zero when all queues are empty. Floating-point ratios are used. P scoring does not include Device KV utilization.

`--car_host_hit_weight` defaults to 0.5 and accepts [0, 1]. It weights Host-only reuse relative to Device reuse as a tunable heuristic, not a measured transfer-time model. A block present on both tiers counts only as Device. All matches require a continuous prefix of complete blocks.

D retains its load-based ordering by selecting the smallest cost:

```text
D score = gpu_cache_usage_perc
          + waiting_requests_num / maximum waiting_requests_num among D candidates
```

D does not receive prefix-hit credit for reserved blocks. Equal-cost candidates rotate priority.

Service can choose an instance but cannot pin its internal DP rank. A multi-DP instance therefore receives the **minimum hit ratio across all its DP ranks**, never the best rank or a sum. A missing or unsynchronized DP makes the instance's hit ratio zero; it can still be selected by the cost formula. CAR checks incarnation identity, or registration time for older engines, to avoid crediting a replacement process with stale cache.

Disconnects, sequence gaps and incomplete snapshots affect cache credit without blocking routing. Each service replica independently subscribes and recovers. The deprecated heartbeat `cache_event` field is ignored, and this revision no longer reads or writes `XLLM:CACHE:` keys.

## Events and recovery semantics

The live protocol has three frames: `topic | uint64 big-endian sequence | MessagePack`. A batch is `[timestamp, events, dp_rank]`; typed event maps include `BlockStored`, `BlockRemoved` and `AllBlocksCleared`, with `gpu`/`cpu` media. Device/Host data is already in this channel; no `motor_pool_endpoint` is needed. Support covers current xLLM flat-text events, not arbitrary vLLM versions or LoRA/multimodal contexts.

The index builds XXH3-128 content keys from full prefix context and tokens, retaining engine-hash mappings for removals and tokenless replica events. A folded 64-bit wire hash is not the old service's XXH3-128 key. Device removal leaves a Host replica intact.

On first connection, gaps, errors or reconnection, the xLLM replay endpoint returns a **current full snapshot**, not arbitrary historical events. A DEALER sends `[empty delimiter, 8-byte starting sequence]`, receives `[empty delimiter, snapshot watermark, MessagePack]`, then waits for a UINT64_MAX/empty-payload terminator. This differs from the live topic envelope and does not imply generic vLLM historical-replay compatibility.

Recovery buffers bounded live deltas, removes duplicates against the watermark and publishes a complete query view atomically. A live full batch containing `AllBlocksCleared + Stored` can also restore synchronization. Sequence gaps, unknown parents, parsing errors, disconnects or resource-limit violations invalidate the affected stream and expose an unsynchronized state instead of stale hits. Periodic snapshots detect silently lost tail events; a quiet publisher alone does not mean a disconnect. Parent-first snapshots skip orphaned suffixes whose parents have left both tiers, while allowing other independent prefixes; a missing parent in incremental events triggers recovery.

## Optional regression checks

ZeroMQ is managed by service's vcpkg manifest. In a prepared build environment:

```bash
mkdir -p build
cd build
cmake .. -DBUILD_TESTING=ON
make -j8
ctest --output-on-failure -R '^(KV(Index|EventDecoder|Directory|CacheQuery)|CacheAwareRouting)'
```

These protocol regressions use TCP XPUB/ROUTER publishers and require no model, NPU or etcd. They cover discovery, tier replicas, contiguous prefixes, DP isolation, snapshot/delta merging, disconnects, process replacement, capacity limits and internal-query validation.
