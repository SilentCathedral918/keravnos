#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nvml.h>

#include <fstream>
#include <iostream>
#include <random>

#define NUM_THREADS 256

namespace py = pybind11;
