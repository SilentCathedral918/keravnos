#pragma once

#include "global.cuh"

template <typename T>
void memory_allocate(T* &ptr, std::size_t alloc_size) {
  void *ptr_ = nullptr;
  cudaError_t err_ = cudaMalloc(&ptr_, alloc_size);
  
  if (err_ != cudaSuccess) {
    py::print("[keravnos cuda error] cudaMalloc() failed:", cudaGetErrorString(err_));
    ptr = nullptr;
    return;
  }
  
  ptr = static_cast<T*>(ptr_);
}

template <typename T>
void memory_deallocate(T* &ptr) {
  cudaError_t err_ = cudaFree(ptr);
  
  if (err_ != cudaSuccess) {
    py::print("[keravnos cuda error] cudaFree() failed: ", cudaGetErrorString(err_));
    return;
  }

  ptr = nullptr;
}

std::unordered_map<std::string, std::size_t> memory_get_gpu_vram(void);

