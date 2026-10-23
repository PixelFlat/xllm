---
title: "Adaptive DSpark"
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

Adaptive DSpark uses confidence feedback and calibrated full-step costs to
select a verification budget. Parameter types and defaults are listed in the
[CLI reference](/en/cli_reference/#speculativeconfig).

The adaptive-pruning flag and JSON key are now `enable_adaptive_target_verify`
instead of `enable_adaptive_spec_prune` or `enable_adaptive_speculative_decode`. Update existing launch
scripts and JSON configurations; the old name is not an alias.

## Supported configuration

The supported adaptive task pipeline uses an NPU GLM DSA Python target, a DSpark
draft with a confidence head, `num_speculative_tokens=5` or `7`, greedy draft sampling,
DP=1/CP=1, ACL graphs, schedule overlap and task pipeline. Graph warmup must be
enabled. Add `--enable_unified_mtp_graph=true` to capture each complete adaptive
round, including the greedy heads, confidence consensus, packing, verification
and accepted-context update. Unified graphs currently require `temperature=0`;
positive or mixed temperatures must use ordinary graphs. Such requests are
rejected before execution when unified graphs are enabled, never silently
decoded greedily. Fixed-width DSpark and other algorithms retain their own
execution contracts.

## Launch example

Add the following settings to **every rank** of an existing multi-rank launch.
This command illustrates rank zero of a 16-rank service; start the remaining
ranks with their own `node_rank` and listening port. Create the output directory
before launch and choose an output file that does not already exist:

```bash
mkdir -p /results
export HCCL_OP_EXPANSION_MODE=AIV
xllm --model=/models/target --draft_model=/models/dspark \
  --backend=llm --model_impl=python --npu_kernel_backend=TORCH \
  --nnodes=16 --node_rank=0 --host=127.0.0.1 --port=8000 \
  --master_node_addr=127.0.0.1:27000 \
  --speculative_algorithm=DSpark --num_speculative_tokens=7 \
  --draft_sampling_mode=greedy --enable_adaptive_target_verify=true \
  --enable_graph=true --python_graph_backend=aclgraph \
  --disable_graph_warmup=false --enable_schedule_overlap=true \
  --enable_task_pipeline=true --enable_unified_mtp_graph=false \
  --dp_size=1 --cp_size=1 --max_seqs_per_batch=16 \
  --acl_graph_decode_batch_size_limit=16 \
  --adaptive_dspark_cost_table_output=/results/dspark-costs.json
```

Send requests with an explicit temperature: `temperature=0` selects greedy
target decoding; a positive value such as `temperature=0.5` selects random
target sampling. Different temperatures can share a batch. Target top-k, top-p
and min-p filtering use the ordinary sampler's semantics. The draft remains
greedy (`--draft_sampling_mode=greedy`); probabilistic drafts are not admitted.

For random targets, the rejection sampler maps request parameters through the
device-side packed-row owners and draws one target token per verification row.
With a deterministic draft this is exact point-mass rejection sampling: matching
draws accept the draft prefix, and the first mismatch supplies the residual
token; an all-matching prefix uses its terminal draw as the bonus. All TP ranks
agree on the draws before acceptance and context-KV publication. Target/draft
forward and adaptive packing retain ACL graph replay; random sampling and
prefix commit execute on the task stream outside the model graph, without
reading the packed boundaries back to Host. Per-request seeds, logprobs,
structured constraints, minimum-length stop masks, token bias and non-neutral penalties remain unsupported
and are rejected before execution rather than silently ignored.

The budget is still selected ahead of the round from eligible lagged confidence;
current target draws never feed back into the same round's clipping decision.
Greedy-draft confidence need not predict stochastic acceptance accurately.
Startup calibration currently uses greedy target requests, so its full-step
costs do not include random-sampling overhead. Random-target correctness does
not imply the same acceptance rate, optimal budgets or performance as greedy
targets; measure them separately.

For Draft5, change only `--num_speculative_tokens=5` and select a new output
file such as `/results/dspark-costs-draft5.json`. The draft generates five
candidates; adaptive verification can select a shorter prefix. The three
device kernels use width-specific compiled specializations, and Slot storage
is sized to the selected width rather than padded to Draft7. The width is
fixed at startup, included in graph binding keys and saved-table compatibility,
and cannot change between replays. Draft5 and Draft7 costs must not be mixed.
Unversioned legacy tables without a `draft_width` field are treated as Draft7;
newly fitted bare tables record their width explicitly.

## Startup calibration

No HTTP calibration client, external fitting command or second service launch
is needed. Startup first warms the graphs, drives the real asynchronous
prefill/decode pipeline, collects in-memory completion windows, fits costs on
CPU and installs the same immutable table on every worker before ready. This
does not add per-round timing Events, profiling D2H copies or file writes to
normal serving. Existing result-readiness Events and asynchronous
result/confidence feedback remain necessary.

### Built-in defaults

With no calibration file or saved table, the built-in settings are:

| Setting | Effective default |
|:--------|:------------------|
| `model` | `additive` |
| `draft_width` | Selected `num_speculative_tokens`; `5` or `7` |
| `request_buckets` | Actual captured physical draft request buckets |
| `target_graph_rows` | All legal captured target tiers, including full width, for each request bucket |
| `contexts` | One interval: prefill 2048 tokens, KV coverage 1–4096 |
| `warmup_steps` | `8` |
| `window_steps` | `32` complete decode intervals per window |
| `windows_per_point` | `3` |
| `token_seed` | `1` |

The default KV upper bound is capped by the model's context limit. For a shorter
context, prefill is reduced to reserve the worst-case generated tokens and both
in-flight slots. If the default sampling steps still do not fit, startup fails
with a request for a smaller explicit configuration. Sampling every captured
bucket can increase startup time; use a JSON file to restrict the grid.

### Custom configuration

To override the defaults, add `--adaptive_dspark_calibration=/configs/dspark.json`.
The width is inferred from the launch configuration; an optional `draft_width`
field in the JSON must match it. For example:

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

### Cost models and coverage

`model` accepts `sps` or `additive`. The SPS model is a one-dimensional
steps-per-second table over physical target graph rows `G`; cost is
`1000 / SPS(G)` milliseconds. The additive model fits full-step seconds as
`bias + alpha(Bcap) + theta(G)`, where `Bcap` is the physical draft request
bucket, not logical HTTP concurrency. These are full-step costs: do not add draft
or commit time again. The additive probe grid must be connected through shared
G values. Request buckets and any optional `target_graph_rows` must be captured;
the grid must retain full-width coverage. Context intervals must not overlap and
must fit the model, cache and worst-case sampling progress.

Unified startup captures at most four target tiers per actual request count
`B`: the first prepared buckets covering `B`, `2B`, `4B` and
`B * (draft_width + 1)`. For example, Draft5 at `B=8` uses `G=8,16,32,48`.
Calibration uses only complete `B/G` bindings captured for both pipeline slots;
an explicit unsupported tier fails before sampling. Transient request sets use
their prewarmed full-width binding and do not count as stable samples. Serving
never captures a missing binding. Split and unified costs are distinct, and
saved bundles validate the unified setting before installation.

Both default and file-based startup log `DSpark adaptive effective configuration`
with the source, all effective settings (including omitted defaults) and actual
captured geometry. No serving request causes profiling, fitting or graph
recapture. A KV range or request bucket without a calibrated profile uses the
already-prewarmed full-width admission policy, not an eager retry.

## Reuse a saved cost table

To reuse a saved bundle on a later compatible launch, replace the calibration
and output flags with `--adaptive_dspark_cost_table=/results/dspark-costs.json`.
The source, saved calibration settings and effective coverage are logged.
Checkpoint/configuration and parallelism/execution compatibility are checked;
saved-table loading skips sampling/fitting, not graph warmup. Random-token
calibration and a low training error do not prove a speedup: compare real-request
TPOT, acceptance and accepted length against a fixed-width baseline.
