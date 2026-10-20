# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exact-budget allocation of confidence-ranked DSpark verification prefixes."""

import tilelang.language as T
from tilelang import tvm

DSPARK_ALLOCATE_VERIFY_PREFIXES_PASS_CONFIGS = {
    "tl.disable_safe_memory_legalize": True,
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}

MAX_DSPARK_BATCH_SIZE = 128
DSPARK_DRAFT_TOKENS = 7


def build_dspark_allocate_verify_prefixes_kernel(max_batch_size: int, device_control: bool = False) -> tvm.tir.PrimFunc:
    """Find the boundary score with vector sort, then resolve its ties exactly.

    Sorting indices are deliberately unused: hardware sort stability is not
    part of the ABI. Position-major counting resolves equal boundary scores
    by position, then request. Only O(B*gamma) UB memory is used, not an NxN
    pairwise ranking matrix. Confidence normalization matches the reference.
    """
    if max_batch_size not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError("max_batch_size must be a supported power-of-two bucket")
    gamma = DSPARK_DRAFT_TOKENS
    batch_tile = ((max_batch_size + 31) // 32) * 32
    sort_size = ((max_batch_size * gamma + 31) // 32) * 32

    @T.prim_func
    def dspark_allocate_verify_prefixes(
        confidence_handle: T.handle,
        widths_handle: T.handle,
        control_handle: T.handle,
        batch_size: T.int32,
        budget: T.int32,
    ):
        confidence = T.match_buffer(confidence_handle, (batch_size * (22 if device_control else gamma),), "float32")
        # Keep sizes as expressions of the explicit INT32 ABI metadata. A raw
        # Var shape is otherwise inferred as another parameter by this backend.
        widths = T.match_buffer(
            widths_handle, (batch_size * 22 if device_control else T.Select(batch_size > 0, batch_size, 0),), "int32"
        )
        control = T.match_buffer(control_handle, (1,), "int64")
        with T.Kernel(1, is_npu=True) as (_, vid), T.Scope("V"):
            if vid == 0:
                width_ub = T.alloc_ub((batch_tile,), "int32")
                input_ub = T.alloc_ub((sort_size,), "float32")
                probability_ub = T.alloc_ub((batch_tile,), "float32")
                product_ub = T.alloc_ub((batch_tile,), "float32")
                survival_ub = T.alloc_ub((sort_size,), "float32")
                sorted_ub = T.alloc_ub((sort_size * 2,), "float32")
                threshold = T.alloc_var("float32", init=0.0)
                probability = T.alloc_var("float32", init=0.0)
                greater_count = T.alloc_var("int32", init=0)
                ties_taken = T.alloc_var("int32", init=0)
                active_rows = T.alloc_var("int32", init=batch_size)
                selected_budget = T.alloc_var("int32", init=budget)
                confidence_offset = T.alloc_var("int32", init=0)
                width_offset = T.alloc_var("int32", init=0)
                if device_control:
                    control_ub = T.alloc_ub((4,), "int64")
                    T.copy(control[0:1], control_ub)
                    T.barrier_all()
                    active_rows = T.Cast("int32", control_ub[0])
                    # The graph capacity is immutable; the live request count
                    # and compact-arena offsets are device-authoritative.
                    selected_budget = T.min(active_rows * gamma, budget - active_rows)
                    confidence_offset = active_rows * 14
                    width_offset = active_rows * 21
                T.tile.fill(width_ub, 1)
                if selected_budget == active_rows * gamma:
                    T.tile.fill(width_ub, gamma + 1)
                elif selected_budget > 0:
                    T.copy(confidence[confidence_offset : confidence_offset + active_rows * gamma], input_ub)
                    T.tile.fill(product_ub, 1.0)
                    T.tile.fill(survival_ub, -1.0)
                    for column in T.serial(gamma):
                        T.tile.fill(probability_ub, 1.0)
                        for row in T.serial(active_rows):
                            probability = input_ub[row * gamma + column]
                            if probability != probability:
                                probability = 0.0
                            probability_ub[row] = T.min(T.max(probability, 0.0), 1.0)
                        T.tile.mul(product_ub, product_ub, probability_ub)
                        for row in T.serial(active_rows):
                            survival_ub[column * max_batch_size + row] = product_ub[row]
                    T.tile.sort(sorted_ub, survival_ub, max_batch_size * gamma)
                    # The sort helper's last merge/copy is asynchronous. Do
                    # not rely on compiler defaults before a scalar UB load.
                    T.barrier_all()
                    threshold = sorted_ub[(selected_budget - 1) * 2]
                    # First count strictly greater scores. This makes the tie
                    # quota independent of the sort's internal tie ordering.
                    for column in T.serial(gamma):
                        for row in T.serial(active_rows):
                            if survival_ub[column * max_batch_size + row] > threshold:
                                greater_count += 1
                    for column in T.serial(gamma):
                        for row in T.serial(active_rows):
                            probability = survival_ub[column * max_batch_size + row]
                            if probability > threshold:
                                width_ub[row] += 1
                            elif probability == threshold and ties_taken < selected_budget - greater_count:
                                width_ub[row] += 1
                                ties_taken += 1
                if active_rows > 0:
                    T.copy(width_ub, widths[width_offset : width_offset + active_rows])

    return dspark_allocate_verify_prefixes
