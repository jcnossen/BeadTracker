#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <curand.h>
#include <curand_kernel.h>
#include <npp.h>
#include <cublas.h>

typedef struct CUstream_st *cudaStream_t;

#define BLKSIZE 16 // thread block size

#define ITEMIDX() \
	int2 pos = make_int2(blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y); \
	if (pos.y > size.y || pos.x > size.x) return; \
	int idx = pos.y*size.x+pos.x;


#ifdef __host__
#define CUDA_BOTH __device__ __host__
#define CUDA_HOST __host__
#else
#define CUDA_BOTH
#define CUDA_HOST
#endif

