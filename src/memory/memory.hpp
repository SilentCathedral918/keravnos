#include "global.hpp"

void memory_allocate(__half* &ptr, const std::size_t alloc_size);
void memory_allocate_from_array(__half* &ptr, const pybind11::array_t<float> input);
void memory_allocate_transformer(
  __half* &ptr,
  const int batch_size = 2,
  const int sequence_length = 2048,
  const int num_dims = 768,
  const int num_layers = 12,
  const int num_heads = 12,
  const int precision_size = 2,
  const bool activation_checkpoint = true
);

void memory_deallocate(__half* &ptr);

std::size_t memory_estimate_transformer_total_vram_usage(
  const int batch_size = 2,
  const int sequence_length = 2048,
  const int num_dims = 768,
  const int num_layers = 12,
  const int num_heads = 12,
  const int precision_size = 2,
  const bool activation_checkpoint = true
);

std::unordered_map<std::string, std::size_t> memory_get_gpu_vram(void);
