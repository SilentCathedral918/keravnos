#include "global.cuh"

typedef struct TransformerHeader {
  __half*       _token_embed;
  __half*       _pos_embed;
  int*          _token_ids;
  __half*       _input_embed;
  __half*       _dropout;

  __half*       _qkv_proj;
  __half*       _qkv_matrix;
  __half*       _qkv_bias;

  __half*       _attn_scores;
  __half*       _context_layer;
  __half*       _out_proj;
  __half*       _out_proj_bias;
  __half*       _output;   

  std::uint32_t _batch_size;
  std::uint32_t _sequence_length;
  std::uint32_t _vocab_size;
  std::uint32_t _num_dims;
  std::uint32_t _num_heads;
  std::uint32_t _type_bytes;

  bool          _allocated;
  std::size_t   _mem_total; 
} TransformerHeader;

void transformer_allocate_memory(
  __half* &ptr, 
  const int batch_size, 
  const int sequence_length, 
  const int vocab_size,
  const int num_dims,
  const int num_heads
);

void transformer_deallocate_memory(__half* &ptr);

void transformer_generate_embedding_weights(__half* &ptr);

void transformer_load_from_file(__half* &out, const char *filepath);
void transformer_save_to_file(__half* &ptr, const char *filepath);

py::array_t<float> transformer_token_embedding(__half* &ptr);
py::array_t<float> transformer_positional_embedding(__half* &ptr);

void transformer_embed_input_tokens(__half* &ptr, const int *token_ids);

void transformer_generate_qkv_projection(__half* &ptr);
void transformer_generate_qkv_bias(__half* &ptr);
void transformer_generate_output_projection(__half* &ptr);
void transformer_generate_output_bias(__half* &ptr);

void transformer_reset_weights(__half* &ptr);

void transformer_causal_self_attention(__half* &ptr, const bool bias, const float dropout, const std::uint64_t seed = 0);

