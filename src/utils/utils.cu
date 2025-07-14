#include "utils/utils.h"

__global__ void _utils_convert_float_to_half(__half *out, const float *in, const int n) {
  int idx_ = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx_ < n)
    out[idx_] = __float2half(in[idx_]);
}

extern "C" void utils_convert_float_to_half(__half *out, float *in, const int n, const int num_threads) {
  _utils_convert_float_to_half<<<(n + num_threads - 1) / num_threads, num_threads>>>(out, in, n);
}
