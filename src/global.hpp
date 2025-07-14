#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <nvml.h>

#include <iostream>

#define NUM_THREADS 256

