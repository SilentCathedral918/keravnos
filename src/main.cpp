#include "memory/memory.hpp"

static __half *keravnos_mem;

void keravnos_activate(
  const int batch_size = 2,
  const int sequence_length = 2048,
  const int num_dims = 768,
  const int num_layers = 12,
  const int num_heads = 12,
  const int precision_size = 2,
  const bool activation_checkpoint = true
) {
  memory_allocate_transformer(
    keravnos_mem, 
    batch_size, 
    sequence_length, 
    num_dims, 
    num_layers, 
    num_heads, 
    precision_size, 
    activation_checkpoint
  );
}

void keravnos_deactivate(void) {
  memory_deallocate(keravnos_mem);
}

pybind11::str keravnos_estimate_vram_usage(
  const int batch_size = 2,
  const int sequence_length = 2048,
  const int num_dims = 768,
  const int num_layers = 12,
  const int num_heads = 12,
  const int precision_size = 2,
  const bool activation_checkpoint = true
) {
  std::size_t vram_usage_ = memory_estimate_transformer_total_vram_usage(
    batch_size,
    sequence_length,
    num_dims,
    num_layers,
    num_heads,
    precision_size,
    activation_checkpoint
  );

  std::stringstream ss_;
  ss_ << "Estimated VRAM usage: ";

  float vram_f_ = static_cast<float>(vram_usage_);
  
  if (vram_usage_ >= (1ULL << 30))
    ss_ << (vram_f_ / (1 << 30)) << " GB";
  else if (vram_usage_ >= (1ULL << 20))
    ss_ << (vram_f_ / (1 << 20)) << " MB";
  else if (vram_usage_ >= (1ULL << 10))
    ss_ << (vram_f_ / (1 << 10)) << " KB";
  else
    ss_ << vram_usage_ << " bytes";

  return pybind11::str(ss_.str());
}

pybind11::dict keravnos_report_gpu_vram(void) {
  pybind11::dict out_;
  std::unordered_map<std::string, std::size_t> stats_ = memory_get_gpu_vram();

  out_["used"] = stats_["Device 0 _ VRAM Used (bytes)"];
  out_["free"] = stats_["Device 0 _ VRAM Free (bytes)"];
  out_["total"] = stats_["Device 0 _ VRAM Total (bytes)"];

  return out_;
}

PYBIND11_MODULE(keravnos, m) {
  m.def(
    "activate", 
    keravnos_activate,
    pybind11::arg("batch_size") = 2,
    pybind11::arg("sequence_length") = 2048,
    pybind11::arg("num_dims") = 768,
    pybind11::arg("num_layers") = 12,
    pybind11::arg("num_heads") = 12,
    pybind11::arg("precision_size") = 2,
    pybind11::arg("activation_checkpoint") = true
  );
  m.def("deactivate", keravnos_deactivate);
  m.def(
    "estimate_vram_usage", 
    keravnos_estimate_vram_usage,
    pybind11::arg("batch_size") = 2,
    pybind11::arg("sequence_length") = 2048,
    pybind11::arg("num_dims") = 768,
    pybind11::arg("num_layers") = 12,
    pybind11::arg("num_heads") = 12,
    pybind11::arg("precision_size") = 2,
    pybind11::arg("activation_checkpoint") = true
  );
  m.def("get_gpu_vram", memory_get_gpu_vram);
}
