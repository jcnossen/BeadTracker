#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <npp.h>
#include <cublas.h>

#ifdef __host__
#define CUDA_BOTH __device__ __host__
#define CUDA_HOST __host__
#else
#define CUDA_BOTH
#define CUDA_HOST
#endif

