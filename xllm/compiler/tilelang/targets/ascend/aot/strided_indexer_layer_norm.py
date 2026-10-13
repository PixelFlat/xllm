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
from compiler.tilelang.common.spec import DispatchField, TilelangKernel, register_kernel

from xllm.python.kernels_npu.tilelang import mtp_prepare_next_draft as vector_impl
from xllm.python.kernels_npu.tilelang import strided_indexer_layer_norm as kernel_impl
from xllm.python.kernels_npu.tilelang.mtp_prepare_next_draft import prepare_vector_only_source
from xllm.python.kernels_npu.tilelang.strided_indexer_layer_norm import (
    STRIDED_INDEXER_PASS_CONFIGS,
    STRIDED_INDEXER_TOKEN_SHAPES,
    build_strided_indexer_layer_norm_kernel,
)

DEPENDENCY_MODULES = (kernel_impl, vector_impl)


@register_kernel
class StridedIndexerLayerNormKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("num_tokens", "int32")]
    SPECIALIZATIONS = [
        {"variant_key": f"tokens_{tokens}", "num_tokens": tokens} for tokens in STRIDED_INDEXER_TOKEN_SHAPES
    ]

    @staticmethod
    def generate_source(num_tokens: int) -> str:
        tilelang.disable_cache()
        kernel = build_strided_indexer_layer_norm_kernel(num_tokens)
        with tilelang.tvm.transform.PassContext(opt_level=3, config=STRIDED_INDEXER_PASS_CONFIGS):
            lowered = tilelang.engine.lower(kernel)
        source = lowered.kernel_source
        if source.count("KERNEL_TYPE_MIX_AIC_1_1") != 1 or any(
            operation in source for operation in ("Mmad(", "CrossCoreSetFlag", "CrossCoreWaitFlag")
        ):
            raise RuntimeError("Strided Indexer normalization must contain only vector work")
        return prepare_vector_only_source(source)
