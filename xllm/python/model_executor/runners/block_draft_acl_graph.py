# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ACL graphs for fixed-width, non-causal DFlash/DSpark draft blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.model_executor.forward_context import AclGraphExecutionState
from xllm.python.model_executor.runners.acl_graph import (
    AclGraphEntry,
    StaticGraphAttentionMetadata,
)
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner

_BlockGraphKey = tuple[object, ...]


class BlockDraftAclGraphRunner(DecodeAclGraphRunner):
    """Capture one graph for each DSpark sequence-count/width shape.

    DSpark presents ``N`` non-causal query rows per request.  The regular ACL
    decode runner intentionally rejects this packed layout because its graph
    metadata has one row per sequence.  This runner keeps one row per request
    in the paged attention metadata and uses device KV lengths to mask the
    changing accepted prefix during replay.
    """

    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
        max_batch: int,
        max_model_len: int,
    ) -> None:
        super().__init__(model, attention_backend, device, max_batch, max_model_len)

    def can_execute(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> bool:
        if self.dp_size != 1 or input_ids.dim() != 1:
            return False
        if not metadata.is_chunked_prefill or metadata.is_prefill or metadata.is_spec_verify:
            return False
        if input_embedding is not None or mtp_topk_indices is not None:
            return False
        q_seq_lens = metadata.q_seq_lens
        block_table = metadata.block_table
        slot_mapping = metadata.slot_mapping
        kv_seq_lens = metadata.kv_seq_lens
        if q_seq_lens is None or block_table is None or slot_mapping is None or kv_seq_lens is None:
            return False
        if q_seq_lens.dim() != 1 or q_seq_lens.numel() == 0:
            return False
        query_widths = q_seq_lens.to(torch.int64).tolist()
        query_width = query_widths[0]
        if query_width <= 0 or any(width != query_width for width in query_widths):
            return False
        sequence_count = len(query_widths)
        if sequence_count > self.max_batch or block_table.shape != (sequence_count, block_table.shape[1]):
            return False
        if input_ids.numel() != sequence_count * query_width:
            return False
        if slot_mapping.dim() != 1 or slot_mapping.numel() != input_ids.numel():
            return False
        if kv_seq_lens.dtype != torch.int32 or kv_seq_lens.shape != (sequence_count,):
            return False
        query_ends = self._query_ends(metadata, query_widths)
        if query_ends is None or query_ends[-1] != input_ids.numel():
            return False
        return block_table.dim() == 2 and block_table.shape[1] > 0

    @staticmethod
    def _query_ends(metadata: AttentionMetadata, query_widths: list[int]) -> list[int] | None:
        host_ends = getattr(metadata, "q_cu_seq_lens_host_values", None)
        if host_ends is None:
            return None
        host_ends = list(host_ends)
        if len(host_ends) == len(query_widths) + 1 and host_ends[0] == 0:
            host_ends = host_ends[1:]
        if len(host_ends) != len(query_widths):
            return None
        expected = []
        total = 0
        for width in query_widths:
            total += width
            expected.append(total)
        return host_ends if host_ends == expected else None

    def warmup(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> _BlockGraphKey:
        self._validate_inputs(input_ids, positions, metadata)
        sequence_count = metadata.q_seq_lens.numel()
        query_width = int(metadata.q_seq_lens[0].item())
        graph_key = self._graph_key(input_ids, metadata, sequence_count, query_width)
        if graph_key in self._graphs:
            return graph_key
        self._prepare_graph_entry(input_ids, positions, metadata, graph_key=graph_key)
        return graph_key

    @staticmethod
    def _graph_key(
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        sequence_count: int,
        query_width: int,
    ) -> _BlockGraphKey:
        block_table = metadata.block_table
        return (
            sequence_count,
            query_width,
            int(block_table.shape[1]),
            input_ids.dtype,
            input_ids.device,
        )

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> None:
        if not self.can_execute(input_ids, metadata):
            raise ValueError("DSpark ACL graph requires fixed-width non-causal block metadata")
        if positions.dim() != 1 or positions.numel() != input_ids.numel():
            raise ValueError("DSpark ACL graph positions must match input_ids")

    def _prepare_graph_entry(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        graph_key: _BlockGraphKey,
    ) -> AclGraphEntry:
        if input_embedding is not None or mtp_topk_indices is not None:
            raise ValueError("DSpark ACL graph does not accept embedding or MTP graph inputs")
        entry = self._graphs.get(graph_key)
        first_capture = entry is None
        if first_capture:
            entry = self._allocate_entry(input_ids, positions, metadata)
            self._graphs[graph_key] = entry
        if self._stream is None:
            self._stream = torch.npu.Stream(device=input_ids.device)
            self._initialize_task_updates()
            self._update_done_event = torch.npu.Event()
        if self._replay_done_event is not None:
            torch.npu.current_stream().wait_event(self._replay_done_event)
        if self._update_done_recorded:
            assert self._update_done_event is not None
            torch.npu.current_stream().wait_event(self._update_done_event)
        self._fill_entry(entry, input_ids, positions, metadata)
        self._prepare_attention(entry, entry.static_metadata)
        if first_capture:
            self._capture(entry, self._stream)
        return entry

    def _allocate_entry(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> AclGraphEntry:
        del positions
        sequence_count = metadata.q_seq_lens.numel()
        query_width = int(metadata.q_seq_lens[0].item())
        block_cols = metadata.block_table.shape[1]
        token_count = sequence_count * query_width
        device = input_ids.device
        page_size = self._logical_page_size
        kv_capacity = block_cols * page_size
        query_ends = self._query_ends(metadata, [query_width] * sequence_count)
        if query_ends is None:
            raise ValueError("DSpark ACL graph requires canonical query ends")
        static_block_table = torch.zeros_like(metadata.block_table, dtype=torch.int32).contiguous()
        static_kv_lens = torch.empty(sequence_count, dtype=torch.int32, device=device)
        static_metadata = StaticGraphAttentionMetadata(
            slot_mapping=torch.zeros(token_count, dtype=metadata.slot_mapping.dtype, device=device),
            paged_kv_indptr=torch.zeros(sequence_count + 1, dtype=torch.int32, device=device),
            paged_kv_indices=torch.zeros(sequence_count * block_cols, dtype=torch.int32, device=device),
            paged_kv_last_page_len=torch.ones(sequence_count, dtype=torch.int32, device=device),
            q_cu_seq_lens=(
                metadata.q_cu_seq_lens.clone()
                if metadata.q_cu_seq_lens is not None
                else torch.tensor(query_ends, dtype=torch.int32, device=device)
            ),
            q_cu_seq_lens_host_values=list(query_ends),
            q_seq_lens=torch.full(
                (sequence_count,),
                query_width,
                dtype=torch.int32,
                device=device,
            ),
            q_seq_lens_host=(
                metadata.q_seq_lens_host.clone() if getattr(metadata, "q_seq_lens_host", None) is not None else None
            ),
            kv_cu_seq_lens=(metadata.kv_cu_seq_lens.clone() if metadata.kv_cu_seq_lens is not None else None),
            kv_seq_lens_host_values=[kv_capacity] * sequence_count,
            block_table=static_block_table,
            kv_seq_lens=static_kv_lens,
            is_chunked_prefill=True,
        )
        static_metadata.prepared_attention_state = self.attention_backend.prepare_metadata(
            static_metadata, device_kv_lengths=True
        )
        entry = AclGraphEntry()
        entry.batch_size = token_count
        entry.graph = None
        entry.static_output = None
        entry.static_input_ids = torch.zeros_like(input_ids)
        entry.static_positions = torch.zeros_like(input_ids, dtype=torch.int32)
        entry.static_input_embedding = None
        entry.static_mtp_topk_indices = None
        entry.static_metadata = static_metadata
        entry.kv_seq_lens_delta = static_kv_lens
        entry.graph_tasks = []
        entry.execution_state = AclGraphExecutionState({})
        entry.replay_logged = False
        return entry

    def _fill_entry(
        self,
        entry: AclGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> None:
        static = entry.static_metadata
        entry.static_input_ids.copy_(input_ids)
        entry.static_positions.copy_(positions.to(torch.int32))
        static.slot_mapping.copy_(metadata.slot_mapping)
        static.block_table.copy_(metadata.block_table.to(torch.int32))
        static.kv_seq_lens.copy_(metadata.kv_seq_lens)
        if metadata.paged_kv_indptr is not None:
            static.paged_kv_indptr.copy_(metadata.paged_kv_indptr)
        if metadata.paged_kv_last_page_len is not None:
            static.paged_kv_last_page_len.copy_(metadata.paged_kv_last_page_len)
        if metadata.paged_kv_indices is not None:
            static.paged_kv_indices.zero_()
            count = min(static.paged_kv_indices.numel(), metadata.paged_kv_indices.numel())
            static.paged_kv_indices[:count].copy_(metadata.paged_kv_indices[:count])
        kv_host = getattr(metadata, "kv_seq_lens_host_values", None)
        if kv_host is not None:
            static.kv_seq_lens_host_values[: len(kv_host)] = list(kv_host)
