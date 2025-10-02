#pragma once

#include "global.cuh"

#ifdef __cplusplus
extern "C" {
#endif

void utils_convert_float_to_half(__half *out, float *in, const int n, const int num_threads = NUM_THREADS);
void utils_convert_half_to_float(float *out, __half *in, const int n, const int num_threads = NUM_THREADS);

void utils_generate_random_half_dim1(__half *out, const int dx, const int min, const int max, const int num_threads = NUM_THREADS);
void utils_generate_random_half_dim2(__half *out, const int dx, const int dy, const int min, const int max, const int num_threads = NUM_THREADS);
void utils_generate_random_half_dim3(__half *out, const int dx, const int dy, const int dz, const int min, const int max, const int num_threads = NUM_THREADS);

void utils_generate_sinusoidal_half(__half *out, const int dx, const int dy, const int base, const int num_threads = NUM_THREADS);

#ifdef __cplusplus
}
#endif
