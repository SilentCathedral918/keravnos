#include "blocks/causal_self_attention.cuh"
#include "neural_network/dropout.cuh"

#ifdef __cplusplus
extern "C" {
#endif 

__global__ void selfattn_qkv_bias(
  __half *qkv_matrix,
  const __half *qkv_bias,
  const int m,
  const int dim_3
) {
  const int idx_ = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_ >= m * dim_3) return;

  qkv_matrix[idx_] = __hadd(qkv_matrix[idx_], qkv_bias[idx_ % dim_3]);
}

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
) {
  const int m_ = 3 * n_dims;
  const int n_ = batch_size * seq_len;
  const int k_ = n_dims;
  const __half alpha_ = __float2half(1.0f);
  const __half beta_  = __float2half(0.0f);

  cublasGemmEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    m_, n_, k_,
    &alpha_,
    qkv_proj, CUDA_R_16F, m_,
    input_embed, CUDA_R_16F, k_,
    &beta_,
    qkv_matrix, CUDA_R_16F, m_,
    CUDA_R_32F,
    CUBLAS_GEMM_DEFAULT 
  );

  if (use_bias) {
    const int total_ = (n_dims * 3) * batch_size * seq_len;
    const int blocks_ = (total_ + NUM_THREADS - 1) / NUM_THREADS;

    selfattn_qkv_bias<<<blocks_, NUM_THREADS>>>(qkv_matrix, qkv_bias, total_, n_dims * 3);
  }
}

__global__ void selfattn_attention_scores(
  __half *attn_scores,
  const __half *q_matrix,
  const __half *k_matrix,
  const int seq_len,
  const int n_heads,
  const int head_dim,
  const float scale,
  const bool use_causal_mask
) {
  const int seq_idxx_ = blockIdx.x;
  const int seq_idxy_ = blockIdx.y;
  const int bh_idx_ = blockIdx.z;
  const int batch_idx_ = bh_idx_ / n_heads;
  const int head_idx_  = bh_idx_ % n_heads;
  const int q_offset_ = ((batch_idx_ * n_heads + head_idx_) * seq_len + seq_idxx_) * head_dim;
  const int k_offset_ = ((batch_idx_ * n_heads + head_idx_) * seq_len + seq_idxy_) * head_dim;

  __half score_ = __float2half(0.0f);

  for (int di_ = 0; di_ < head_dim; ++di_) {
    score_ = __hadd(score_, __hmul(q_matrix[q_offset_ + di_], k_matrix[k_offset_ + di_]));
  }

  score_ = __hdiv(score_, __float2half(scale));
  
  const int attn_offset_ = ((batch_idx_ * n_heads + head_idx_) * seq_len + seq_idxx_) * seq_len + seq_idxy_;
  attn_scores[attn_offset_] = score_;

  if (use_causal_mask && (seq_idxy_ > seq_idxx_)) {
    attn_scores[attn_offset_] = __float2half(-HALF_INFINITY);
  }
}

__global__ void selfattn_softmax_dropout(
  __half *attn_scores,
  const int seq_len,
  const int n_heads,
  const float dropout_rate,
  const std::uint64_t seed
) {
  const int seq_idxx_ = blockIdx.x;
  const int bh_idx_  = blockIdx.y; 
  const int seq_idxy_ = threadIdx.x;
  const int offset_ = (bh_idx_ * seq_len + seq_idxx_) * seq_len;

  // find max
  __half max_val_ = __float2half(-HALF_INFINITY);
  for (int idx_ = 0; idx_ < seq_len; ++idx_) {
    max_val_ = __hmax(max_val_, attn_scores[offset_ + idx_]);
  } 

  // compute exp (score - max)
  float sum_ = 0.0f;
  for (int idx_ = 0; idx_ < seq_len; ++idx_) {
    float exp_ = expf(__half2float(attn_scores[offset_ + idx_]) - __half2float(max_val_));
    attn_scores[offset_ + idx_] = __float2half(exp_);
    sum_ += exp_;
  }

  // normalise
  float inv_sum_ = 1.0f / sum_;
  attn_scores[offset_ + seq_idxy_] = __hmul(attn_scores[offset_ + seq_idxy_], __float2half(inv_sum_));

  // dropout
  curandStatePhilox4_32_10_t state_;
  curand_init(seed, offset_ + seq_idxy_, 0, &state_);
  
  const float keep_prob_ = 1.0f - dropout_rate;
  const float rand_ = curand_uniform(&state_);

  if (rand_ >= keep_prob_) {
    attn_scores[offset_ + seq_idxy_] = __float2half(0.0f);
  }
  else {
    attn_scores[offset_ + seq_idxy_] = __hdiv(attn_scores[offset_ + seq_idxy_], __float2half(keep_prob_));
  }
}


__global__ void selfattn_attention_weighted_values(
  __half *output,
  const __half *attn_probs,
  const __half *v_matrix,
  const int seq_len,
  const int n_heads,
  const int head_dim
) {
  const int ts_idx_ = blockIdx.x;
  const int bh_idx_ = blockIdx.y;
  const int dim_idx_ = threadIdx.x;
  const int attn_offset_ = (bh_idx_ * seq_len + ts_idx_) * seq_len;
  const int value_offset_ = bh_idx_ * seq_len * head_dim;
  const int out_offset_ = (bh_idx_ * seq_len + ts_idx_) * head_dim;

  __half accum_ = __float2half(0.0f);

  for (int idx_ = 0; idx_ < seq_len; ++idx_) {
    __half attn_ = attn_probs[attn_offset_ + idx_];
    __half val_ = v_matrix[value_offset_ + idx_ * head_dim + dim_idx_];

    accum_ = __hadd(accum_, __hmul(attn_, val_));
  }

  output[out_offset_ + dim_idx_] = accum_;
}

__global__ void selfattn_projection_bias(
  __half *output,
  const __half *proj_bias,
  const int n_dims,
  const int rows
) {
  const int idx_ = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_ >= rows * n_dims) return;

  const int dim_ = idx_ % n_dims;
  output[idx_] = __hadd(output[idx_], proj_bias[dim_]);
}

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
) {
  const int m_ = n_dims;
  const int n_ = batch_size * seq_len;
  const int k_ = n_heads * head_dim;
  const __half alpha_ = __float2half(1.0f);
  const __half beta_  = __float2half(0.0f);

  cublasGemmEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_T,
    m_, n_, k_,
    &alpha_,
    proj_weights, CUDA_R_16F, m_,
    context, CUDA_R_16F, k_,
    &beta_,
    output, CUDA_R_16F, m_,
    CUDA_R_32F,
    CUBLAS_GEMM_DEFAULT
  );

  if (use_bias) {
    const int total_ = batch_size * seq_len * n_dims;
    const int blocks_ = (total_ + NUM_THREADS - 1) / NUM_THREADS;

    selfattn_projection_bias<<<blocks_, NUM_THREADS>>>(output, proj_bias, n_dims, batch_size * seq_len);
  }
}

#ifdef __cplusplus
}
#endif 

