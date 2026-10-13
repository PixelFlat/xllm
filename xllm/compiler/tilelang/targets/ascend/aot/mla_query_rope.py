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

from xllm.python.kernels_npu.tilelang import mla_query_rope as kernel_impl
from xllm.python.kernels_npu.tilelang import mtp_prepare_next_draft as vector_impl
from xllm.python.kernels_npu.tilelang.mla_query_rope import (
    MLA_QUERY_ROPE_PASS_CONFIGS,
    MLA_QUERY_ROPE_TOKEN_SHAPES,
    build_mla_query_rope_kernel,
)
from xllm.python.kernels_npu.tilelang.mtp_prepare_next_draft import prepare_vector_only_source

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, vector_impl)


@register_kernel
class MlaQueryRopeKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("num_tokens", "int32")]
    SPECIALIZATIONS = [
        {"variant_key": f"tokens_{tokens}", "num_tokens": tokens} for tokens in MLA_QUERY_ROPE_TOKEN_SHAPES
    ]

    @staticmethod
    def generate_source(num_tokens: int) -> str:
        tilelang.disable_cache()
        kernel = build_mla_query_rope_kernel(num_tokens)
        with tilelang.tvm.transform.PassContext(opt_level=3, config=MLA_QUERY_ROPE_PASS_CONFIGS):
            lowered = tilelang.engine.lower(kernel)
        source = lowered.kernel_source
        if source.count("KERNEL_TYPE_MIX_AIC_1_1") != 1 or any(
            operation in source for operation in ("Mmad(", "CrossCoreSetFlag", "CrossCoreWaitFlag")
        ):
            raise RuntimeError("MLA query RoPE must contain only vector work without cross-core dependencies")
        return prepare_vector_only_source(source)
