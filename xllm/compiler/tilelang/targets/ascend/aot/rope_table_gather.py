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


import tilelang

from xllm.python.kernels_npu.tilelang import mtp_prepare_next_draft as vector_impl
from xllm.python.kernels_npu.tilelang import rope_table_gather as kernel_impl
from xllm.python.kernels_npu.tilelang.mtp_prepare_next_draft import prepare_vector_only_source
from xllm.python.kernels_npu.tilelang.rope_table_gather import (
    ROPE_TABLE_GATHER_PASS_CONFIGS,
    ROPE_TABLE_GATHER_TOKEN_SHAPES,
    build_rope_table_gather_kernel,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, vector_impl)


@register_kernel
class RopeTableGatherKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("num_tokens", "int32"), DispatchField("position_bits", "int32")]
    SPECIALIZATIONS = [
        {"variant_key": f"tokens_{tokens}_positions_{bits}", "num_tokens": tokens, "position_bits": bits}
        for tokens in ROPE_TABLE_GATHER_TOKEN_SHAPES
        for bits in (32, 64)
    ]

    @staticmethod
    def generate_source(num_tokens: int, position_bits: int) -> str:
        tilelang.disable_cache()
        kernel = build_rope_table_gather_kernel(num_tokens, position_bits)
        with tilelang.tvm.transform.PassContext(opt_level=3, config=ROPE_TABLE_GATHER_PASS_CONFIGS):
            lowered = tilelang.engine.lower(kernel)
        source = lowered.kernel_source
        if source.count("KERNEL_TYPE_MIX_AIC_1_1") != 1 or any(
            operation in source for operation in ("Mmad(", "CrossCoreSetFlag", "CrossCoreWaitFlag")
        ):
            raise RuntimeError("RoPE gather must contain only vector work")
        return prepare_vector_only_source(source)
