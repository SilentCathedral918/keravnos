#include "utils/utils.cuh"

// ---------------------------------------- implementation ---------------------------------------- //

__global__ void _utils_convert_float_to_half(__half *out, const float *in, const int n) {
  int idx_ = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_ >= n) return;
  
  out[idx_] = __float2half(in[idx_]);
}

__global__ void _utils_convert_half_to_float(float *out, const __half *in, const int n) {
  int idx_ = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_ >= n) return;
  
  out[idx_] = __half2float(in[idx_]);
}

__global__ void _utils_generate_random_half(__half *out, const int n, const int min, const int max, const std::uint64_t seed) {
  int idx_ = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_ >= n) return;

  if (min == max) {
    out[idx_] = __float2half(static_cast<float>(min));
    return;
  }

  curandState state_;
  curand_init(seed, idx_, 0, &state_);

  float val_ = min + curand_uniform(&state_) * (max - min);
  out[idx_] = __float2half(val_);
}

__global__ void _utils_generate_sinusoidal_half(__half *out, const int n, const int n_dims, const int base) {
  int idx_ = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_ >= n) return;

  int pos_ = idx_ / n_dims;
  int i_   = idx_ % n_dims;

  float exp_ = -2.0f * (i_ / 2) / static_cast<float>(n_dims);
  float angle_ = pos_ * __powf(static_cast<float>(base), exp_);

  float val_ = (i_ % 2 == 0) ? __sinf(angle_) : __cosf(angle_);
  out[idx_] = __float2half(val_);
}


// ---------------------------------------- interface ---------------------------------------- //

#ifdef __cplusplus
extern "C" {
#endif

void utils_convert_float_to_half(__half *out, float *in, const int n, const int num_threads) {
  _utils_convert_float_to_half<<<(n + num_threads - 1) / num_threads, num_threads>>>(out, in, n);
}

void utils_convert_half_to_float(float *out, __half *in, const int n, const int num_threads) {
  _utils_convert_half_to_float<<<(n + num_threads - 1) / num_threads, num_threads>>>(out, in, n);
}


void utils_generate_random_half_dim1(__half *out, const int dx, const int min, const int max, const int num_threads) {
  const int n_ = dx;
  std::uint64_t seed_ = static_cast<std::uint64_t>(std::random_device{}());
  _utils_generate_random_half<<<(n_ + num_threads - 1) / num_threads, num_threads>>>(out, n_, min, max, seed_);
}

void utils_generate_random_half_dim2(__half *out, const int dx, const int dy, const int min, const int max, const int num_threads) {
  const int n_ = dx * dy;
  std::uint64_t seed_ = static_cast<std::uint64_t>(std::random_device{}());
  _utils_generate_random_half<<<(n_ + num_threads - 1) / num_threads, num_threads>>>(out, n_, min, max, seed_);
}

void utils_generate_random_half_dim3(__half *out, const int dx, const int dy, const int dz, const int min, const int max, const int num_threads) {
  const int n_ = dx * dy * dz;
  std::uint64_t seed_ = static_cast<std::uint64_t>(std::random_device{}());
  _utils_generate_random_half<<<(n_ + num_threads - 1) / num_threads, num_threads>>>(out, n_, min, max, seed_);
}


void utils_generate_sinusoidal_half(__half *out, const int dx, const int dy, const int base, const int num_threads) {
  const int n_ = dx * dy;
  _utils_generate_sinusoidal_half<<<(n_ + num_threads - 1) / num_threads, num_threads>>>(out, n_, dy, base);
}

#ifdef __cplusplus
}
#endif

