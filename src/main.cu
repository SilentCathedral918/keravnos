#include "transformer/transformer.cuh"

static __half* transformer_weights = nullptr;

void keravnos_activate(int batch_size, int seq_len, int vocab_size, int n_dims, const bool verbose) {
  if (transformer_weights != nullptr) {
    if (verbose) py::print("[keravnos error] Transformer is already activated.");
    return;
  }

  // allocate memory
  if (verbose) py::print("[keravnos] Activating transformer...");
  transformer_allocate_memory(transformer_weights, batch_size, seq_len, vocab_size, n_dims);
  if (transformer_weights == nullptr) {
    if (verbose) py::print("[keravnos error] Transformer activation failed.");
    return;
  }

  // generate embedding weights
  transformer_generate_embedding_weights(transformer_weights, seq_len, vocab_size, n_dims);
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

  // reset vram utility for keravnos
  cudaDeviceReset();
  if (verbose) py::print("[keravnos] Transformer deactivation completed.");
}

py::array_t<float> keravnos_token_embedding(const bool verbose) {
  if (transformer_weights == nullptr) {
    if (verbose) py::print("[keravnos error] Transformer is not activated.");
    py::array_t<float>();
  }

  return transformer_token_embedding(transformer_weights);
}

py::array_t<float> keravnos_position_embedding(const bool verbose) {
  if (transformer_weights == nullptr) {
    if (verbose) py::print("[keravnos error] Transformer is not activated.");
    py::array_t<float>();
  }

  return transformer_positional_embedding(transformer_weights);
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
}
