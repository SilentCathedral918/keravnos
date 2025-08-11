#include "transformer/transformer.cuh"

static __half* transformer_weights = nullptr;

void keravnos_activate(int batch_size, int seq_len, int vocab_size, int n_dims, int n_heads, const bool verbose) {
  if (transformer_weights != nullptr) {
    if (verbose) py::print("[keravnos error] Transformer is already activated.");
    return;
  }

  // allocate memory
  if (verbose) py::print("[keravnos] Activating transformer...");
  transformer_allocate_memory(transformer_weights, batch_size, seq_len, vocab_size, n_dims, n_heads);
  if (transformer_weights == nullptr) {
    if (verbose) py::print("[keravnos error] Transformer activation failed.");
    return;
  }

  // generate embedding weights
  transformer_generate_embedding_weights(transformer_weights);
  if (verbose) py::print("[keravnos] Transformer generated embedding weights.");
  
  if (verbose) py::print("[keravnos] Transformer activation completed.");
}

void keravnos_activate_from_file(const std::string filepath, const bool verbose) {
  if (transformer_weights != nullptr) {
    if (verbose) py::print("[keravnos error] Transformer is already activated.");
    return;
  }

  // load from file
  transformer_load_from_file(transformer_weights, filepath.c_str());
  if (transformer_weights == nullptr) {
    if (verbose) py::print("[keravnos error] Transformer activation from file failed.");
    return;
  }

  if (verbose) py::print("[keravnos] Transformer activation from file completed.");
}

void keravnos_deactivate(const std::string save_filepath, const bool verbose) {
  if (transformer_weights == nullptr) {
    if (verbose) py::print("[keravnos error] Transformer is not activated.");
    return;
  }

  // save to filepath location had filepath string not being empty
  if (save_filepath == "") {
    if (verbose) py::print("[keravnos] save_filepath is empty. Skipping transformer weights savings...");
  }
  else {
    transformer_save_to_file(transformer_weights, save_filepath.c_str());
  }

  // reset vram utility
  cudaDeviceReset();
  transformer_weights = nullptr;
  if (verbose) py::print("[keravnos] Transformer deactivation completed.");
}

py::array_t<float> keravnos_token_embedding(const bool verbose) {
  if (transformer_weights == nullptr) {
    if (verbose) py::print("[keravnos error] Transformer is not activated.");
    return py::array_t<float>();
  }

  return transformer_token_embedding(transformer_weights);
}

py::array_t<float> keravnos_position_embedding(const bool verbose) {
  if (transformer_weights == nullptr) {
    if (verbose) py::print("[keravnos error] Transformer is not activated.");
    return py::array_t<float>();
  }

  return transformer_positional_embedding(transformer_weights);
}

void keravnos_embed_input_token_ids(py::array_t<int> token_ids, const bool verbose) {
  if (transformer_weights == nullptr) {
    if (verbose) py::print("[keravnos error] Transformer is not activated.");
    return;
  }

  // force copy to contiguous
  token_ids = token_ids.attr("copy")();

  py::buffer_info buf_info_ = token_ids.request();
  const int *ids_ = reinterpret_cast<int *>(buf_info_.ptr);
  const std::size_t num_ids_ = buf_info_.size;

  TransformerHeader header_;
  cudaMemcpy(&header_, transformer_weights, sizeof(TransformerHeader), cudaMemcpyDeviceToHost);

  if (num_ids_ != header_._batch_size * header_._sequence_length) {
    if (verbose) py::print("[keravnos error] Expected", header_._batch_size * header_._sequence_length, "token IDs but got", num_ids_, "token IDs instead.");
    return;
  }

  transformer_embed_input_tokens(transformer_weights, ids_);
}

// ------------------------------ interface ------------------------------ //

PYBIND11_MODULE(keravnos, m) {
  m.def(
    "activate",
    &keravnos_activate,
    py::arg("batch_size") = 2,
    py::arg("sequence_length") = 2048,
    py::arg("vocab_size") = 48000,
    py::arg("num_dims") = 768,
    py::arg("num_heads") = 12,
    py::arg("verbose") = false
  );

  m.def(
    "activate_from_file",
    &keravnos_activate_from_file,
    py::arg("filepath") = "",
    py::arg("verbose") = false
  );

  m.def(
    "deactivate",
    &keravnos_deactivate,
    py::arg("save_filepath") = "",
    py::arg("verbose") = false
  );

  m.def(
    "get_token_embedding",
    &keravnos_token_embedding,
    py::arg("verbose") = false
  );
  
  m.def(
    "get_positional_embedding",
    &keravnos_position_embedding,
    py::arg("verbose") = false
  );

  m.def(
    "embed_input_token_ids",
    &keravnos_embed_input_token_ids,
    py::arg("token_ids"),
    py::arg("verbose") = false
  );
}
