#include "global.cuh"

typedef struct TransformerHeader {
  __half*       _token_embed;
  __half*       _pos_embed;

  std::uint32_t _batch_size;
  std::uint32_t _sequence_length;
  std::uint32_t _vocab_size;
  std::uint32_t _num_dims;
  std::uint32_t _type_bytes;

  bool          _allocated;
  std::size_t   _mem_total; 
} TransformerHeader;

void transformer_allocate_memory(
  __half* &ptr, 
  const int batch_size, 
  const int sequence_length, 
  const int vocab_size,
  const int num_dims
);

void transformer_deallocate_memory(__half* &ptr);

void transformer_generate_embedding_weights(
  __half* &ptr,
  const int sequence_length,
  const int vocab_size, 
  const int num_dims
);

void transformer_load_from_file(__half* &out, const char *filepath);
void transformer_save_to_file(__half* &ptr, const char *filepath);

py::array_t<float> transformer_token_embedding(__half* &ptr);
py::array_t<float> transformer_positional_embedding(__half* &ptr);
