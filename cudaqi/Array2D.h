/*
CUDA Tracker Array2D.
Contains the CUDA code to apply per-pixel calculations to arrays and sum them by reduce.
*/
#include <cuda_runtime.h>
#include <thrust/functional.h>

#include "cuda_shared_mem.h"

void throwCudaError(cudaError_t err);

inline void cudaCheck(cudaError_t err) {
	if (err != cudaSuccess) 
		throwCudaError(err);
}


template<typename T>
class pixel_value
{
public:
	__host__ __device__ T operator()(const T& value, const unsigned int x, const unsigned int y) {
		return value;
	}
};

// Center of mass X
template<typename T>
class pixel_COM_x
{
public:
	__host__ __device__ T operator()(const T& value, const unsigned int x, const unsigned int y) {
		return value * x;
	}
};

// Center of mass Y
template<typename T>
class pixel_COM_y
{
public:
	__host__ __device__ T operator()(const T& value, const unsigned int x, const unsigned int y) {
		return value * y;
	}
};


template<typename T, typename TBinaryFunc, unsigned int blockSize, typename TPixelOp>
__global__ void reduceArray2D_k(const T* src, size_t spitch, T* out, size_t width, size_t outputWidth)
{
	T* sdata = SharedMemory<T>();

	int tid = threadIdx.x;
	int xpos = threadIdx.x + blockIdx.x * blockSize;

	TPixelOp pixel_op;
	sdata[threadIdx.x] = xpos < width ? pixel_op(src[ spitch * blockIdx.y + xpos ], xpos, blockIdx.y) : 0;

	__syncthreads();

	const int warpStart = 16;
	TBinaryFunc reduce_op;
	for (unsigned int k=blockSize/2;k>warpStart;k>>=1) {
		if(tid < k) {
			sdata[tid] = reduce_op(sdata[tid], sdata[tid+k]);
		}
		__syncthreads();
	}

	if (tid < warpStart)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile T *smem = sdata;
		if (blockSize >=  64) { smem[tid] = reduce_op((T)smem[tid], (T)smem[tid + 32]); }
		if (blockSize >=  32) { smem[tid] = reduce_op((T)smem[tid], (T)smem[tid + 16]); }
		if (blockSize >=  16) { smem[tid] = reduce_op((T)smem[tid], (T)smem[tid +  8]); }
		if (blockSize >=   8) { smem[tid] = reduce_op((T)smem[tid], (T)smem[tid +  4]); }
		if (blockSize >=   4) { smem[tid] = reduce_op((T)smem[tid], (T)smem[tid +  2]); }
		if (blockSize >=   2) { smem[tid] = reduce_op((T)smem[tid], (T)smem[tid +  1]); }
	}

	// write result 
	if (tid==0) out[blockIdx.y*outputWidth + blockIdx.x] = sdata[0];
}


/* 
2D Array using CUDA pitched device memory 
*/
template<typename T>
class Array2D
{
public:
	size_t pitch, w, h;
	T* data;

	Array2D(size_t w,size_t h) {
		this->w = w;
		this->h = h;
		cudaCheck(cudaMallocPitch(&data, &pitch, w*sizeof(T), h));
	}

	~Array2D() {
		if (data)
			cudaFree(data);
	}

	Array2D(size_t w, size_t h, const T* host_data, size_t host_pitch=0) {
		this->w = w;
		this->h = h;
		cudaCheck(cudaMallocPitch(&data, &pitch, w*sizeof(T), h));
		if (host_pitch==0) host_pitch = w*sizeof(T);
		cudaCheck(cudaMemcpy2D(data, pitch, host_data, host_pitch, w*sizeof(T), h, cudaMemcpyHostToDevice));
	}

	T* allocAndCopyToHost() {
		T* hostData = new T[w*h];
		cudaCheck(cudaMemcpy2D(hostData, w*sizeof(T), data, pitch, sizeof(T)*w, h, cudaMemcpyDeviceToHost));
		return hostData;
	}

	T sum()
	{
		return reduce_array_2D<thrust::plus<T>, pixel_value<T> >::apply(*this);
	}

	T momentX()
	{
		return reduce_array_2D<thrust::plus<T>, pixel_COM_x<T> >::apply(*this);
	}
	T momentY()
	{
		return reduce_array_2D<thrust::plus<T>, pixel_COM_y<T> >::apply(*this);
	}

	template<typename TBinOp = thrust::plus<T>, typename TPixelOp = pixel_value<T> >
	class reduce_array_2D
	{
	public:

		static T apply(Array2D<T> & a)
		{
			const int blockSize = 32;
			const int cpuThreshold = 32;
			dim3 nThreads(blockSize, 1,1);

			// Allocate memory for per-block results
			int nBlockX = (a.w + blockSize - 1) / blockSize;
			Array2D<T> blockResults1(nBlockX * a.h, 1);
			Array2D<T> blockResults2( (nBlockX * a.h + blockSize - 1) / blockSize, 1);
			Array2D<T> *output = &blockResults1, *input = &a;

			// Reduce image
			int width = a.w;
			bool first=true;
			dim3 nBlocks(nBlockX, input->h, 1);
			do {
				if (first) {
					reduceArray2D_k<T, TBinOp, blockSize, TPixelOp > <<<nBlocks, nThreads, blockSize*sizeof(T)>>> (input->data, input->pitch/sizeof(T), output->data, width, nBlocks.x);
					first=false;
				} else {
					reduceArray2D_k<T, TBinOp, blockSize, pixel_value<T> > <<<nBlocks, nThreads, blockSize*sizeof(T)>>> (input->data, input->pitch/sizeof(T), output->data, width, nBlocks.x);
				}
				width = nBlocks.x*nBlocks.y;
				nBlocks = dim3( (width + blockSize - 1)/blockSize, 1 ,1);
				std::swap(output, input);
				if (output == &a) output = &blockResults2;
			} while (width > cpuThreshold);

			// Copy to host memory. Block results are now in 'input' due to last std::swap
			T* results = input->allocAndCopyToHost();
			T resultValue = 0.0f;
			TBinOp binary_op;
			for (int x=0;x<width;x++) {
				resultValue = binary_op(resultValue, results[x]);
			}
			delete results;
			return resultValue;
		}
	};


};



