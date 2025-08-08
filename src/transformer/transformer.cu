#include "memory/memory.cuh"
#include "utils/utils.cuh"
#include "transformer/transformer.cuh"

void transformer_allocate_memory(
  __half* &ptr, 
  const int batch_size, 
  const int sequence_length,
  const int vocab_size, 
  const int num_dims
) {
  const std::size_t required_ = 
    sizeof(TransformerHeader)                   /* transformer header */    +
    vocab_size * num_dims * sizeof(__half)      /* token embedding */       +
    sequence_length * num_dims * sizeof(__half) /* positional embedding */
  ;

  py::print("[keravnos cuda] Allocating", required_, "bytes...");
  memory_allocate(ptr, required_);
  if (!ptr) return;

  TransformerHeader staging_;
  staging_._token_embed = reinterpret_cast<__half *>(reinterpret_cast<std::uintptr_t>(ptr) + sizeof(TransformerHeader));
  staging_._pos_embed = reinterpret_cast<__half *>(reinterpret_cast<std::uintptr_t>(ptr) + sizeof(TransformerHeader) + (vocab_size * num_dims));
  staging_._batch_size = batch_size;
  staging_._sequence_length = sequence_length;
  staging_._vocab_size = vocab_size;
  staging_._num_dims = num_dims;
  staging_._type_bytes = sizeof(__half);
  staging_._mem_total = required_;
  staging_._allocated = true;
  
  cudaMemcpy(ptr, &staging_, sizeof(TransformerHeader), cudaMemcpyHostToDevice);

  py::print("\n[keravnos cuda] Transformer Memory Allocated");
  py::print("-----------------------------------");
  py::print("Total Bytes          :", required_, "bytes");
  py::print("Batch Size           :", batch_size);
  py::print("Sequence Length      :", sequence_length);
  py::print("Vocab Size           :", vocab_size);
  py::print("Embedding Dim        :", num_dims);
  py::print("Header Size          :", sizeof(TransformerHeader), "bytes");
  py::print("Token Embedding      :", vocab_size * num_dims * sizeof(__half), "bytes");
  py::print("Positional Embedding :", sequence_length * num_dims * sizeof(__half), "bytes");
}

void transformer_deallocate_memory(__half* &ptr) {
  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);
  
  memory_deallocate(ptr);
  py::print("[keravnos cuda] Transformer memory dellocated.");
}

void transformer_generate_embedding_weights(
  __half* &ptr,
  const int sequence_length,
  const int vocab_size, 
  const int num_dims
) {
  const int token_embed_limit_ = vocab_size * num_dims;

  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);

  utils_generate_random_half_dim2(header_._token_embed, vocab_size, num_dims, token_embed_limit_ * -1, token_embed_limit_);  
  utils_generate_sinusoidal_half(header_._pos_embed, sequence_length, num_dims, 10000);
}

void transformer_load_from_file(__half* &out, const char *filepath) {
  std::ifstream in_(filepath, std::ios::binary | std::ios::ate);
  if (!in_.is_open()) {
    py::print("[keravnos cuda error] Failed to open file:", filepath);
    return;
  }
  
  std::streamsize size_ = in_.tellg();
  in_.seekg(0, std::ios::beg);

  std::vector<char> vec_data_(size_);
  in_.read(vec_data_.data(), size_);
  in_.close();

  TransformerHeader* header_ = reinterpret_cast<TransformerHeader*>(vec_data_.data());
  
  py::print("[keravnos cuda] Allocating", header_->_mem_total, "bytes...");
  memory_allocate(out, header_->_mem_total);
  if (!out) return;
  
  cudaMemcpy(out, vec_data_.data(), header_->_mem_total, cudaMemcpyHostToDevice);
  py::print("\n[keravnos cuda] Transformer Memory Allocated");
  py::print("-----------------------------------");
  py::print("Total Bytes          :", header_->_mem_total, "bytes");
  py::print("Batch Size           :", header_->_batch_size);
  py::print("Sequence Length      :", header_->_sequence_length);
  py::print("Vocab Size           :", header_->_vocab_size);
  py::print("Embedding Dim        :", header_->_num_dims);
  py::print("Header Size          :", sizeof(TransformerHeader), "bytes");
  py::print("Token Embedding      :", header_->_vocab_size * header_->_num_dims * sizeof(__half), "bytes");
  py::print("Positional Embedding :", header_->_sequence_length * header_->_num_dims * sizeof(__half), "bytes");
}

void transformer_save_to_file(__half* &ptr, const char *filepath) {
  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);
  
  std::vector<char> vec_data_(header_._mem_total);
  cudaMemcpy(vec_data_.data(), ptr, header_._mem_total, cudaMemcpyDeviceToHost);

  std::ofstream out_(filepath, std::ios::binary);
  out_.write(vec_data_.data(), vec_data_.size());

  py::print("[keravnos cuda] Transformer weights saved to filepath:", filepath);
  out_.close();
}

py::array_t<float> transformer_token_embedding(__half* &ptr) {
  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);

  std::size_t num_elems_ = header_._vocab_size * header_._num_dims;
  std::vector<float> vec_out_(num_elems_);
  std::size_t mem_free_ = memory_get_gpu_vram()["Device 0 _ VRAM Free (bytes)"];
  std::size_t mem_required_ = sizeof(float) * num_elems_;
  std::size_t mem_padding_ = 32 * 1024 * 1024; // 32 MB padding

  if (mem_free_ >= mem_required_ + mem_padding_) {
    float *arr_out_ = nullptr;
    memory_allocate(arr_out_, sizeof(float) * num_elems_);
    utils_convert_half_to_float(arr_out_, header_._token_embed, num_elems_);
    cudaMemcpy(vec_out_.data(), arr_out_, sizeof(float) * num_elems_, cudaMemcpyDeviceToHost);
    memory_deallocate(arr_out_);
  }  
  else {
    std::vector<__half> vec_half_(num_elems_);
    cudaMemcpy(vec_half_.data(), header_._token_embed, sizeof(__half) * num_elems_, cudaMemcpyDeviceToHost);
    
    for (std::size_t i = 0; i < num_elems_; ++i)
      vec_out_[i] = __half2float(vec_half_[i]);
  }

  return py::array_t<float>(
    {header_._vocab_size, header_._num_dims},
    {sizeof(float) * header_._num_dims, sizeof(float)},
    vec_out_.data()
  );
}

py::array_t<float> transformer_positional_embedding(__half* &ptr) {
  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);

  std::size_t num_elems_ = header_._sequence_length * header_._num_dims;
  std::vector<float> vec_out_(num_elems_);
  std::size_t mem_free_ = memory_get_gpu_vram()["Device 0 _ VRAM Free (bytes)"];
  std::size_t mem_required_ = sizeof(float) * num_elems_;
  std::size_t mem_padding_ = 32 * 1024 * 1024; // 32 MB padding

  if (mem_free_ >= mem_required_ + mem_padding_) {
    float *arr_out_ = nullptr;
    memory_allocate(arr_out_, sizeof(float) * num_elems_);
    utils_convert_half_to_float(arr_out_, header_._pos_embed, num_elems_);
    cudaMemcpy(vec_out_.data(), arr_out_, sizeof(float) * num_elems_, cudaMemcpyDeviceToHost);
    memory_deallocate(arr_out_);
  }  
  else {
    std::vector<__half> vec_half_(num_elems_);
    cudaMemcpy(vec_half_.data(), header_._pos_embed, sizeof(__half) * num_elems_, cudaMemcpyDeviceToHost);
    
    for (std::size_t i = 0; i < num_elems_; ++i)
      vec_out_[i] = __half2float(vec_half_[i]);
  }

  return py::array_t<float>(
    {header_._sequence_length, header_._num_dims},
    {sizeof(float) * header_._num_dims, sizeof(float)},
    vec_out_.data()
  );
}