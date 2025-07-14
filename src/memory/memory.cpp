#include "memory/memory.hpp"
#include "utils/utils.h"

void memory_allocate(__half* &ptr, const std::size_t alloc_size) {
  cudaError_t err_ = cudaMalloc(reinterpret_cast<void **>(&ptr), alloc_size);
  if (err_ != cudaSuccess) {
    std::cerr << "cudaMalloc() failed: " << cudaGetErrorString(err_) << std::endl;
    return;
  }
}

void memory_allocate_from_array(__half* &ptr, const pybind11::array_t<float> input) {
  pybind11::buffer_info buf_info_ = input.request();
  __half *buf_addr_ = reinterpret_cast<__half *>(buf_info_.ptr);
  std::vector<__half> half_(buf_info_.size);
  float* input_ = static_cast<float*>(buf_info_.ptr);
  cudaError_t err_;
  
  err_ = cudaMalloc(reinterpret_cast<void **>(&ptr), sizeof(__half) * buf_info_.size);
  if (err_ != cudaSuccess) {
    std::cerr << "cudaMalloc() failed: " << cudaGetErrorString(err_) << std::endl;
    return;
  }
  
  float *d_in_;
  err_ = cudaMalloc(&d_in_, sizeof(float) * buf_info_.size);
  if (err_ != cudaSuccess) {
    std::cerr << "cudaMalloc() failed: " << cudaGetErrorString(err_) << std::endl;
    return;
  }
  
  err_ = cudaMemcpy(d_in_, static_cast<float *>(buf_info_.ptr), sizeof(float) * buf_info_.size, cudaMemcpyHostToDevice);
  if (err_ != cudaSuccess) {
    std::cerr << "cudaMemcpy() failed: " << cudaGetErrorString(err_) << std::endl;
    return;
  }

  utils_convert_float_to_half(ptr, d_in_, static_cast<int>(buf_info_.size));
  
  std::cout << "[cuda] Allocated and copied " << buf_info_.size << " floats." << std::endl;
  cudaFree(d_in_);
}

void memory_allocate_transformer(
  __half* &ptr,
  const int batch_size,
  const int sequence_length,
  const int num_dims,
  const int num_layers,
  const int num_heads,
  const int precision_size,
  const bool activation_checkpoint 
) {
  const std::size_t req_vram_ = memory_estimate_transformer_total_vram_usage(
    batch_size,
    sequence_length,
    num_dims,
    num_layers,
    num_heads,
    precision_size,
    activation_checkpoint
  );
  std::unordered_map<std::string, std::size_t> gpu_vram_ = memory_get_gpu_vram();
  const std::size_t available_vram_ = gpu_vram_["Device 0 _ VRAM Free (bytes)"];

  if (available_vram_ < req_vram_)
    pybind11::print(
      "Not enough VRAM. Required: ", req_vram_ / (1024 * 1024 * 1024.0), 
      " GB, Available: ", available_vram_ / (1024 * 1024 * 1024.0), " GB."
    );
  else
    memory_allocate(ptr, req_vram_);
}


void memory_deallocate(__half* &ptr) {
  cudaError_t err_ = cudaFree(ptr);
  if (err_ != cudaSuccess) {
    std::cerr << "cudaFree() failed: " << cudaGetErrorString(err_) << std::endl;
    return;
  }
}


std::size_t memory_estimate_transformer_total_vram_usage(
  const int batch_size,
  const int sequence_length,
  const int num_dims,
  const int num_layers,
  const int num_heads,
  const int precision_size,
  const bool activation_checkpoint
) {
  const int intermediate_ = 4 * num_dims;
  const int opt_mul_ = 2;

  // activation
  std::size_t activation_ = num_layers * batch_size * sequence_length * (6 * num_dims + 4 * num_heads * num_heads + 4 * intermediate_);
  if (activation_checkpoint) activation_ = static_cast<std::size_t>(activation_ * 0.4f);
  std::size_t mem_activation_ = activation_ * precision_size;

  // parameters
  std::size_t params_ = num_layers * (12 * num_dims * num_dims + 8 * num_dims * intermediate_);
  std::size_t mem_params_ = params_ * precision_size;

  // gradients
  std::size_t mem_gradient_ = mem_params_;

  // optimiser
  std::size_t mem_opt_ = mem_params_ * opt_mul_;

  // miscellaneous weights (layer norms, bias, etc)
  std::size_t misc_ = num_layers * (2 * num_dims + intermediate_);
  std::size_t mem_misc_ = misc_ * precision_size * (2 + opt_mul_);
  
  // runtime total (with buffers and overhead)
  std::size_t mem_raw_total_ = mem_activation_ + mem_params_ + mem_gradient_ + mem_opt_ + mem_misc_;
  std::size_t mem_overhead_runtime_ = static_cast<std::size_t>(mem_raw_total_ * 0.1);
  std::size_t mem_overhead_cuda_ = static_cast<std::size_t>(mem_raw_total_ * 0.05);
  std::size_t mem_total_ = mem_raw_total_ + mem_overhead_runtime_ + mem_overhead_cuda_;

  return mem_total_;
}

std::unordered_map<std::string, std::size_t> memory_get_gpu_vram(void) {
  nvmlReturn_t res_;
  std::uint32_t num_devices_;
  nvmlDevice_t device_;
  std::unordered_map<std::string, std::size_t> out_;

  res_ = nvmlInit();
  if (res_ != NVML_SUCCESS) {
    std::cerr << "nvmlInit() failed: " << nvmlErrorString(res_) << std::endl;
    return {};
  }

  res_ = nvmlDeviceGetCount(&num_devices_);
  if (res_ != NVML_SUCCESS) {
    std::cerr << "nvmlDeviceGetCount() failed: " << nvmlErrorString(res_) << std::endl;
    nvmlShutdown();
    return {};
  }

  for (std::size_t i = 0; i < num_devices_; ++i) {
    res_ = nvmlDeviceGetHandleByIndex(i, &device_);
    if (res_ != NVML_SUCCESS) {
      std::cerr << "nvmlDeviceGetHandleByIndex() failed: " << nvmlErrorString(res_) << std::endl;
      continue;
    }

    nvmlMemory_t mem_;
    res_ = nvmlDeviceGetMemoryInfo(device_, &mem_);
    if (res_ != NVML_SUCCESS) {
      std::cerr << "nvmlDeviceGetMemoryInfo() failed: " << nvmlErrorString(res_) << std::endl;
      continue;
    }

    std::stringstream ss_;
    
    ss_ << "Device " << i << " _ VRAM Used (bytes)";
    out_[ss_.str()] = static_cast<std::size_t>(mem_.used);
    ss_.str(""); ss_.clear();

    ss_ << "Device " << i << " _ VRAM Free (bytes)";
    out_[ss_.str()] = static_cast<std::size_t>(mem_.free);
    ss_.str(""); ss_.clear();

    ss_ << "Device " << i << " _ VRAM Total (bytes)";
    out_[ss_.str()] = static_cast<std::size_t>(mem_.total);
    ss_.str(""); ss_.clear();
  }

  nvmlShutdown();
  return out_;
}


