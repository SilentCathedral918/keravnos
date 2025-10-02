#include "memory/memory.cuh"
#include "utils/utils.cuh"
#include "transformer/transformer.cuh"
#include "neural_network/embedding.cuh"
#include "blocks/causal_self_attention.cuh"

void transformer_allocate_memory(
  __half* &ptr, 
  const int batch_size, 
  const int sequence_length,
  const int vocab_size, 
  const int num_dims,
  const int num_heads
) {
  if (num_dims % num_heads != 0) {
    py::print("[keravnos error] Number of dimensions must be divisible by number of heads.");
    return;
  }

  const std::size_t required_ = 
    sizeof(TransformerHeader)                                                           /* transformer header */      +
    vocab_size * num_dims * sizeof(__half)                                              /* token embedding */         +
    sequence_length * num_dims * sizeof(__half)                                         /* positional embedding */    +
    batch_size * sequence_length * sizeof(int)                                          /* token ids */               +    
    batch_size * sequence_length * num_dims * sizeof(__half)                            /* input token vector */      +    
    batch_size * num_heads * sequence_length * sequence_length * sizeof(__half)         /* dropout mask */            +
    num_dims * (3 * num_dims) * sizeof(__half)                                          /* QKV projection weights */  +
    batch_size * sequence_length * (3 * num_dims) * sizeof(__half)                      /* QKV matrix */              +
    (3 * num_dims) * sizeof(__half)                                                     /* QKV bias */                +
    batch_size * num_heads * sequence_length * sequence_length * sizeof(__half)         /* attention scores */        +
    batch_size * num_heads * sequence_length * (num_dims / num_heads) * sizeof(__half)  /* context layer */           +
    num_dims * num_dims * sizeof(__half)                                                /* out projection */          +
    num_dims * sizeof(__half)                                                           /* out projection bias */     +
    batch_size * sequence_length * num_dims * sizeof(__half)                            /* output */
  ;

  py::print("[keravnos cuda] Allocating", required_, "bytes...");
  memory_allocate(ptr, required_);
  if (!ptr) return;

  TransformerHeader staging_;
  std::uintptr_t base_ = reinterpret_cast<std::uintptr_t>(ptr);
  std::size_t offset_ = sizeof(TransformerHeader);
  
  offset_ = ALIGN_OFFSET(offset_, 256);
  staging_._token_embed = reinterpret_cast<__half *>(base_ + offset_);
  offset_ += vocab_size * num_dims * sizeof(__half);

  offset_ = ALIGN_OFFSET(offset_, 256);
  staging_._pos_embed = reinterpret_cast<__half *>(base_ + offset_);
  offset_ += sequence_length * num_dims * sizeof(__half);

  offset_ = ALIGN_OFFSET(offset_, 256);
  staging_._token_ids = reinterpret_cast<int *>(base_ + offset_);
  offset_ += batch_size * sequence_length * sizeof(int);
  
  offset_ = ALIGN_OFFSET(offset_, 256);
  staging_._input_embed = reinterpret_cast<__half *>(base_ + offset_);
  offset_ += batch_size * sequence_length * num_dims * sizeof(__half);
  
  offset_ = ALIGN_OFFSET(offset_, 256);
  staging_._dropout = reinterpret_cast<__half *>(base_ + offset_);
  offset_ += batch_size * num_heads * sequence_length * sequence_length * sizeof(__half);
  
  offset_ = ALIGN_OFFSET(offset_, 256);
  staging_._qkv_proj = reinterpret_cast<__half *>(base_ + offset_);
  offset_ += num_dims * (3 * num_dims) * sizeof(__half);
  
  offset_ = ALIGN_OFFSET(offset_, 256);
  staging_._qkv_matrix = reinterpret_cast<__half *>(base_ + offset_);
  offset_ += batch_size * sequence_length * (3 * num_dims) * sizeof(__half);

  offset_ = ALIGN_OFFSET(offset_, 256);
  staging_._qkv_bias = reinterpret_cast<__half *>(base_ + offset_);
  offset_ += (3 * num_dims) * sizeof(__half);

  offset_ = ALIGN_OFFSET(offset_, 256);
  staging_._attn_scores = reinterpret_cast<__half *>(base_ + offset_);
  offset_ += batch_size * num_heads * sequence_length * sequence_length * sizeof(__half);

  offset_ = ALIGN_OFFSET(offset_, 256);
  staging_._context_layer = reinterpret_cast<__half *>(base_ + offset_);
  offset_ += batch_size * num_heads * sequence_length * (num_dims / num_heads) * sizeof(__half);

  offset_ = ALIGN_OFFSET(offset_, 256);
  staging_._out_proj = reinterpret_cast<__half *>(base_ + offset_);
  offset_ += num_dims * num_dims * sizeof(__half);

  offset_ = ALIGN_OFFSET(offset_, 256);
  staging_._out_proj_bias = reinterpret_cast<__half *>(base_ + offset_);
  offset_ += num_dims * sizeof(__half);

  offset_ = ALIGN_OFFSET(offset_, 256);
  staging_._output = reinterpret_cast<__half *>(base_ + offset_);
  offset_ += batch_size * sequence_length * num_dims * sizeof(__half);

  staging_._batch_size = batch_size;
  staging_._sequence_length = sequence_length;
  staging_._vocab_size = vocab_size;
  staging_._num_dims = num_dims;
  staging_._num_heads = num_heads;
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
  py::print("Number of Heads      :", num_heads);
  py::print("Header Size          :", sizeof(TransformerHeader), "bytes");
  py::print("Token Embedding      :", vocab_size * num_dims * sizeof(__half), "bytes");
  py::print("Positional Embedding :", sequence_length * num_dims * sizeof(__half), "bytes");
  py::print("Token IDs            :", batch_size * sequence_length * sizeof(int), "bytes");
  py::print("Input Embedding      :", batch_size * sequence_length * num_dims * sizeof(__half), "bytes");
  py::print("Dropout Mask         :", batch_size * num_heads * sequence_length * sequence_length * sizeof(__half), "bytes");
}

void transformer_deallocate_memory(__half* &ptr) {
  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);
  
  memory_deallocate(ptr);
  py::print("[keravnos cuda] Transformer memory dellocated.");
}

void transformer_generate_embedding_weights(__half* &ptr) {
  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);
  
  const int token_embed_limit_ = header_._vocab_size * header_._num_dims;

  utils_generate_random_half_dim2(header_._token_embed, header_._vocab_size, header_._num_dims, token_embed_limit_ * -1, token_embed_limit_);  
  utils_generate_sinusoidal_half(header_._pos_embed, header_._sequence_length, header_._num_dims, 10000);
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

void transformer_embed_input_tokens(__half* &ptr, const int *token_ids) {
  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);
  
  // copy token ids to device
  cudaMemcpy(header_._token_ids, token_ids, sizeof(int) * header_._batch_size * header_._sequence_length, cudaMemcpyHostToDevice);

  // embed token ids to input embedding
  embedding_input_vector(ptr);
}

void transformer_generate_qkv_projection(__half* &ptr) {
  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);
  
  const int n_dims_ = header_._num_dims;
  const int qkv_size_ = n_dims_ * (3 * n_dims_);
  
  utils_generate_random_half_dim2(header_._qkv_proj, n_dims_, n_dims_ * 3, -qkv_size_, qkv_size_);
}

void transformer_generate_qkv_bias(__half* &ptr) {
  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);

  const int bias_dim_ = 3 * header_._num_dims;
  utils_generate_random_half_dim1(
    header_._qkv_bias, 
    bias_dim_, 
    -bias_dim_, bias_dim_
  );
}

void transformer_generate_output_projection(__half* &ptr) {
  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);

  const int n_dims_ = header_._num_dims;
  const int out_proj_size_ = n_dims_ * n_dims_;

  utils_generate_random_half_dim2(
    header_._out_proj,
    n_dims_, n_dims_,
    -out_proj_size_, out_proj_size_
  );
}

void transformer_generate_output_bias(__half* &ptr) {
  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);

  const int n_dims_ = header_._num_dims;
  utils_generate_random_half_dim1(
    header_._out_proj_bias,
    n_dims_,
    -n_dims_, n_dims_
  );
}

void transformer_reset_weights(__half* &ptr) {
  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);

  // QKV bias
  cudaMemset(header_._qkv_bias, 0, 3 * header_._num_dims * sizeof(__half));

  // output weights
  cudaMemset(
    header_._output, 
    0, 
    header_._batch_size * header_._sequence_length * header_._num_dims * sizeof(__half)
  );
}

void transformer_causal_self_attention(__half* &ptr, const bool bias, const float dropout, const std::uint64_t seed) {
  TransformerHeader header_;
  cudaMemcpy(&header_, ptr, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);
  
  const int batch_size_ = header_._batch_size;
  const int seq_len_ = header_._sequence_length;
  const int n_dims_ = header_._num_dims;
  const int n_heads_ = header_._num_heads;
  const int head_dim_ = n_dims_ / n_heads_;
  const int qkv_stride_ = batch_size_ * seq_len_ * n_dims_;
  const __half *q_matrix_ = header_._qkv_matrix;
  const __half *k_matrix_ = q_matrix_ + qkv_stride_;
  const __half *v_matrix_ = k_matrix_ + qkv_stride_;

  cublasHandle_t handle_;
  cublasStatus_t stat_ = cublasCreate(&handle_);
  if (stat_ != CUBLAS_STATUS_SUCCESS) {
		py::print("\n[keravnos cuda error] Failed to create CUBLAS.");
		return;
	}

  // QKV projection
  selfattn_qkv_projection(
    header_._qkv_matrix, 
    handle_, 
    header_._input_embed, 
    header_._qkv_proj, header_._qkv_bias,
    n_dims_, batch_size_, seq_len_, 
    bias
  );

  // attention score
  dim3 grid_attn_(seq_len_, seq_len_, batch_size_ * n_heads_);
  selfattn_attention_scores<<<grid_attn_, 1>>>(
    header_._attn_scores,
    q_matrix_, k_matrix_,
    seq_len_, n_heads_, head_dim_, sqrtf(static_cast<float>(head_dim_)),
    true
  );

  // softmax + dropout
  dim3 grid_softmax_(seq_len_, batch_size_ * n_heads_);
  selfattn_softmax_dropout<<<grid_softmax_, seq_len_>>>(
    header_._attn_scores,
    seq_len_, n_heads_, dropout, seed
  );

  // weighted sum of values
  dim3 grid_wv_(seq_len_, batch_size_ * n_heads_);
  selfattn_attention_weighted_values<<<grid_wv_, head_dim_>>>(
    header_._context_layer, header_._attn_scores, v_matrix_,
    seq_len_, n_heads_, head_dim_
  );

  // final projection
  selfattn_output_projection(
    header_._output,
    handle_,
    header_._context_layer, header_._out_proj, header_._out_proj_bias,
    batch_size_, seq_len_, n_dims_, head_dim_, n_heads_,
    bias
  );

  stat_ = cublasDestroy(handle_);
  if (stat_ != CUBLAS_STATUS_SUCCESS) {
		py::print("\n[keravnos cuda error] Failed to destroy CUBLAS.");
		return;
	}
}

