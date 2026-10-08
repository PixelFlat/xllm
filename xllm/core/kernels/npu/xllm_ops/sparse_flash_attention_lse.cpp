/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <glog/logging.h>
#include <torch/library.h>

#include <optional>
#include <string>
#include <string_view>
#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

constexpr int64_t kDim0 = 0;
constexpr int64_t kDim1 = 1;
constexpr int64_t kDim2 = 2;
constexpr int64_t kDim3 = 3;

void check_sparse_flash_attention_lse_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& sparse_indices,
    int64_t sparse_block_size,
    const std::string& layout_query,
    const std::string& layout_kv) {
  TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
  TORCH_CHECK(key.numel() > 0, "Tensor key is empty.");
  TORCH_CHECK(value.numel() > 0, "Tensor value is empty.");
  TORCH_CHECK(sparse_indices.numel() > 0, "Tensor sparse_indices is empty.");
  TORCH_CHECK(
      query.dtype() == torch::kFloat16 || query.dtype() == torch::kBFloat16,
      "query should be FLOAT16 or BFLOAT16.");
  TORCH_CHECK(key.dtype() == query.dtype(),
              "key's dtype should be equal to query's dtype.");
  TORCH_CHECK(value.dtype() == query.dtype(),
              "value's dtype should be equal to query's dtype.");
  TORCH_CHECK(sparse_indices.dtype() == torch::kInt32,
              "sparse_indices should be INT32.");
  TORCH_CHECK(sparse_block_size > 0,
              "sparse_block_size should be greater than 0, actual ",
              sparse_block_size,
              ".");
  TORCH_CHECK(layout_query == "BSND" || layout_query == "TND",
              "The layout of query only support BSND and TND, but got ",
              layout_query);
  TORCH_CHECK(!layout_kv.empty(), "layout_kv should not be empty.");
}

void check_sparse_flash_attention_lse_output(const torch::Tensor& query,
                                             const torch::Tensor& output,
                                             const std::string& layout_query) {
  CHECK_EQ(query.dim(), layout_query == "TND" ? 3 : 4);
  CHECK_EQ(output.device(), query.device())
      << "attention output must use the query device.";
  CHECK_EQ(output.dtype(), query.dtype())
      << "attention output dtype must match query dtype.";
  CHECK(output.is_contiguous()) << "attention output must be contiguous.";
  CHECK(output.sizes() == query.sizes())
      << "attention output shape must match query shape for layout "
      << layout_query;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
construct_sparse_flash_attention_lse_outputs(const torch::Tensor& query,
                                             const torch::Tensor& key,
                                             const std::string& layout_query,
                                             const std::string& layout_kv,
                                             bool return_softmax_lse) {
  at::SmallVector<int64_t, 8> output_size;
  if (layout_query == "TND") {
    TORCH_CHECK(query.dim() == 3,
                "When the layout of query is TND, the query dimension must be "
                "3, but got ",
                query.dim());
    output_size = {query.size(kDim0), query.size(kDim1), query.size(kDim2)};
  } else {
    TORCH_CHECK(query.dim() == 4,
                "When the layout of query is BSND, the query dimension must "
                "be 4, but got ",
                query.dim());
    output_size = {query.size(kDim0),
                   query.size(kDim1),
                   query.size(kDim2),
                   query.size(kDim3)};
  }

  torch::Tensor attention_output =
      torch::empty(output_size, query.options().dtype(query.dtype()));
  at::SmallVector<int64_t, 8> softmax_size;
  if (return_softmax_lse) {
    if (query.dim() == 3) {
      const int64_t kv_head_num =
          layout_kv == "PA_BSND" ? key.size(kDim2) : key.size(kDim1);
      softmax_size = {
          kv_head_num,
          query.size(kDim0),
          query.size(kDim1) / kv_head_num,
      };
    } else {
      softmax_size = {
          query.size(kDim0),
          key.size(kDim2),
          query.size(kDim1),
          query.size(kDim2) / key.size(kDim2),
      };
    }
  } else {
    softmax_size = {0};
  }

  torch::Tensor softmax_max =
      torch::empty(softmax_size, query.options().dtype(torch::kFloat32));
  torch::Tensor softmax_sum =
      torch::empty(softmax_size, query.options().dtype(torch::kFloat32));
  return {attention_output, softmax_max, softmax_sum};
}

void run_sparse_flash_attention_lse(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& sparse_indices,
    const std::optional<torch::Tensor>& block_table,
    const std::optional<torch::Tensor>& actual_seq_lengths_query,
    const std::optional<torch::Tensor>& actual_seq_lengths_kv,
    const std::optional<torch::Tensor>& query_rope,
    const std::optional<torch::Tensor>& key_rope,
    double scale_value,
    int64_t sparse_block_size,
    const std::string& layout_query,
    const std::string& layout_kv,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t attention_mode,
    bool return_softmax_lse,
    const torch::Tensor& attention_output,
    const torch::Tensor& softmax_max,
    const torch::Tensor& softmax_sum) {
  char* query_layout_ptr = const_cast<char*>(layout_query.c_str());
  char* kv_layout_ptr = const_cast<char*>(layout_kv.c_str());
  EXEC_NPU_CMD(aclnnSparseFlashAttentionLse,
               query,
               key,
               value,
               sparse_indices,
               block_table,
               actual_seq_lengths_query,
               actual_seq_lengths_kv,
               query_rope,
               key_rope,
               scale_value,
               sparse_block_size,
               query_layout_ptr,
               kv_layout_ptr,
               sparse_mode,
               pre_tokens,
               next_tokens,
               attention_mode,
               return_softmax_lse,
               attention_output,
               softmax_max,
               softmax_sum);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sparse_flash_attention_lse(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& sparse_indices,
    const std::optional<torch::Tensor>& block_table,
    const std::optional<torch::Tensor>& actual_seq_lengths_query,
    const std::optional<torch::Tensor>& actual_seq_lengths_kv,
    const std::optional<torch::Tensor>& query_rope,
    const std::optional<torch::Tensor>& key_rope,
    double scale_value,
    int64_t sparse_block_size,
    std::string_view layout_query,
    std::string_view layout_kv,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t attention_mode,
    bool return_softmax_lse) {
  std::string layout_query_str = std::string(layout_query);
  std::string layout_kv_str = std::string(layout_kv);
  check_sparse_flash_attention_lse_inputs(query,
                                          key,
                                          value,
                                          sparse_indices,
                                          sparse_block_size,
                                          layout_query_str,
                                          layout_kv_str);
  auto outputs = construct_sparse_flash_attention_lse_outputs(
      query, key, layout_query_str, layout_kv_str, return_softmax_lse);
  torch::Tensor attention_output = std::get<0>(outputs);
  torch::Tensor softmax_max = std::get<1>(outputs);
  torch::Tensor softmax_sum = std::get<2>(outputs);
  run_sparse_flash_attention_lse(query,
                                 key,
                                 value,
                                 sparse_indices,
                                 block_table,
                                 actual_seq_lengths_query,
                                 actual_seq_lengths_kv,
                                 query_rope,
                                 key_rope,
                                 scale_value,
                                 sparse_block_size,
                                 layout_query_str,
                                 layout_kv_str,
                                 sparse_mode,
                                 pre_tokens,
                                 next_tokens,
                                 attention_mode,
                                 return_softmax_lse,
                                 attention_output,
                                 softmax_max,
                                 softmax_sum);
  return {attention_output, softmax_max, softmax_sum};
}

torch::Tensor sparse_flash_attention_lse_out(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& sparse_indices,
    const std::optional<torch::Tensor>& block_table,
    const std::optional<torch::Tensor>& actual_seq_lengths_query,
    const std::optional<torch::Tensor>& actual_seq_lengths_kv,
    const std::optional<torch::Tensor>& query_rope,
    const std::optional<torch::Tensor>& key_rope,
    double scale_value,
    int64_t sparse_block_size,
    std::string_view layout_query,
    std::string_view layout_kv,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t attention_mode,
    bool return_softmax_lse,
    torch::Tensor& attention_output) {
  CHECK(!return_softmax_lse)
      << "sparse_flash_attention_lse_out only supports output without LSE";
  std::string layout_query_str = std::string(layout_query);
  std::string layout_kv_str = std::string(layout_kv);
  check_sparse_flash_attention_lse_inputs(query,
                                          key,
                                          value,
                                          sparse_indices,
                                          sparse_block_size,
                                          layout_query_str,
                                          layout_kv_str);
  check_sparse_flash_attention_lse_output(
      query, attention_output, layout_query_str);
  torch::Tensor empty_lse_max =
      torch::empty({0}, query.options().dtype(torch::kFloat32));
  torch::Tensor empty_lse_sum =
      torch::empty({0}, query.options().dtype(torch::kFloat32));
  run_sparse_flash_attention_lse(query,
                                 key,
                                 value,
                                 sparse_indices,
                                 block_table,
                                 actual_seq_lengths_query,
                                 actual_seq_lengths_kv,
                                 query_rope,
                                 key_rope,
                                 scale_value,
                                 sparse_block_size,
                                 layout_query_str,
                                 layout_kv_str,
                                 sparse_mode,
                                 pre_tokens,
                                 next_tokens,
                                 attention_mode,
                                 false,
                                 attention_output,
                                 empty_lse_max,
                                 empty_lse_sum);
  return attention_output;
}

}  // namespace xllm::kernel::npu
