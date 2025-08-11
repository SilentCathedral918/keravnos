#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nvml.h>

#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>

#define NUM_THREADS 256

#define ALIGN_OFFSET(offset, n) (((offset + (n - 1)) / n) * n)

namespace py = pybind11;
