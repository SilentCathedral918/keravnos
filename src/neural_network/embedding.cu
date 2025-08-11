#include "neural_network/embedding.cuh"
#include "transformer/transformer.cuh"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void _embedding_input_vector(
  __half *out,
  const int *token_ids,
  const __half *token_embed,
  const __half *pos_embed,
  const int vocab_size,
  const int n_dims,
  const int seq_len
) {
  const int b_ = blockIdx.x;
  const int t_ = threadIdx.x;
 
  if (t_ >= seq_len) return;

  const int token_id_ = token_ids[b_ * seq_len + t_];
  if(token_id_ >= vocab_size) return;

  for (int d_ = 0; d_ < n_dims; ++d_) {
    const int embed_idx_ = token_id_ * n_dims + d_;
    const int pos_idx_ = t_ * n_dims + d_;
    const int out_idx_ = (b_ * seq_len + t_) * n_dims + d_;
    const __half token_ = token_embed[embed_idx_];
    const __half pos_ = pos_embed[pos_idx_];

    out[out_idx_] = __hadd(token_, pos_);
  }
} 

void embedding_input_vector(__half* &ptr) {
  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);
  
  _embedding_input_vector<<<header_._batch_size, header_._sequence_length>>>(
    header_._input_embed, 
    header_._token_ids, 
    header_._token_embed, 
    header_._pos_embed, 
    header_._vocab_size, 
    header_._num_dims, 
    header_._sequence_length
  );
}

#ifdef __cplusplus
}
#endif

