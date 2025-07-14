#include "global.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void utils_convert_float_to_half(__half *out, float *in, const int n, const int num_threads = NUM_THREADS);

#ifdef __cplusplus
}
#endif
