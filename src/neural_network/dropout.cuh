#include "global.cuh"
#include "transformer/transformer.cuh"

#ifdef __cplusplus
extern "C" {
#endif 

void dropout_mul_inplace(__half *buf, const float factor, const int n, const std::uint64_t seed = 0, const int num_threads = NUM_THREADS);

#ifdef __cplusplus
}
#endif 

