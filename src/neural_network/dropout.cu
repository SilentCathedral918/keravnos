#include "neural_network/dropout.cuh"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void _dropout_mul_inplace(__half *out, const float factor, const int n, const std::uint64_t seed) {
  int idx_ = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_ >= n) return;

  float prob_ = 1.0f - factor;
  if (prob_ <= 0.0f) { 
    out[idx_] = __float2half(0.0f); 
    return; 
  }

  curandStatePhilox4_32_10_t state_;
  curand_init(seed, idx_, 0, &state_);

  __half val_ = out[idx_];
    if (curand_uniform(&state_) >= prob_) {
        out[idx_] = __float2half(0.0f);
    } else {
        out[idx_] = __hdiv(__hmul(val_, __float2half(1.0f)), __float2half(prob_));
    }

}

void dropout_mul_inplace(__half *buf, const float factor, const int n, const std::uint64_t seed, const int num_threads) {
  std::uint64_t seed_ = (seed == 0) ? static_cast<std::uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()) : seed;
  _dropout_mul_inplace<<<(n + num_threads - 1) / num_threads, num_threads>>>(buf, factor, n, seed_);
}

#ifdef __cplusplus
}
#endif




