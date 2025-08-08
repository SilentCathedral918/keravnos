#include "memory/memory.cuh"
#include "utils/utils.cuh"


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
