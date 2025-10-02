#pragma once

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

#include <cublasLt.h>
#include <cublas_v2.h>

#define NUM_THREADS 256

#define ALIGN_OFFSET(offset, n) (((offset + (n - 1)) / n) * n)

#define HALF_INFINITY 0x7c00

namespace py = pybind11;
