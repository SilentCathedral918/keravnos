#pragma once

#include "global.cuh"
#include "transformer/transformer.cuh"

#ifdef __cplusplus
extern "C" {
#endif 

void dropout_mul_inplace(__half *buf, const float factor, const int n, const std::uint64_t seed = 0, const int num_threads = NUM_THREADS);
__global__ void dropout_apply_kernel(__half *buf, const float prob, const int n, const std::uint64_t seed = 0);

#ifdef __cplusplus
}
#endif 

__device__ __forceinline__ __half dropout_inline(const __half val, const float prob, curandStatePhilox4_32_10_t *state);

