#pragma once

#include "global.cuh"

#ifdef __cplusplus
extern "C" {
#endif 

void selfattn_qkv_projection(
  __half *qkv_matrix, 
  cublasHandle_t handle,
  const __half *input_embed,
  const __half *qkv_proj,
  const __half *qkv_bias, 
  const int n_dims, 
  const int batch_size, 
  const int seq_len, 
  const bool use_bias 
);

__global__ void selfattn_attention_scores(
  __half *attn_scores,
  const __half *q_matrix,
  const __half *k_matrix,
  const int seq_len,
  const int n_heads,
  const int head_dim,
  const float scale,
  const bool use_causal_mask
);

__global__ void selfattn_softmax_dropout(
  __half *attn_scores,
  const int seq_len,
  const int n_heads,
  const float dropout_rate,
  const std::uint64_t seed
);

__global__ void selfattn_attention_weighted_values(
  __half *output,
  const __half *attn_probs,
  const __half *v_matrix,
  const int seq_len,
  const int n_heads,
  const int head_dim
);

void selfattn_output_projection(
  __half *output,
  cublasHandle_t handle,
  const __half *context,
  const __half *proj_weights,
  const __half *proj_bias,
  const int batch_size,
  const int seq_len,
  const int n_dims,
  const int head_dim,
  const int n_heads,
  const bool use_bias
);

#ifdef __cplusplus
}
#endif 
