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
__global__ void reduceArray2D_k(const T* src, size_t spitch, T* out, size_t width)
{
	T* sdata = SharedMemory<T>();

	int tid = threadIdx.x;
	int xpos = threadIdx.x + blockIdx.x * blockSize;

	TPixelOp pixel_op;
	sdata[threadIdx.x] = xpos < width ? pixel_op(src[ spitch * blockIdx.y + xpos ], xpos, blockIdx.y) : 0;

	__syncthreads();

	TBinaryFunc reduce_op;
	for (unsigned int k=blockSize/2;k>32;k>>=1) {
		if(tid < k) {
			sdata[tid] = reduce_op(sdata[tid], sdata[tid+k]);
		}
		__syncthreads();
	}

	if (tid < 32)
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
	if (tid==0) out[blockIdx.y*gridDim.x + blockIdx.x] = sdata[0];
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

	Array2D() {
		pitch=w=h=0; data=0;
	}

	Array2D(size_t w, size_t h, const T* host_data, size_t host_pitch=0) {
		this->w = w;
		this->h = h;
		cudaCheck(cudaMallocPitch(&data, &pitch, w*sizeof(T), h));
		if (host_pitch==0) host_pitch = w*sizeof(T);
		cudaCheck(cudaMemcpy2D(data, pitch, host_data, host_pitch, w*sizeof(T), h, cudaMemcpyHostToDevice));
	}

	void init(int width, int height)
	{
		w = width;
		h = height;
		cudaCheck(cudaMallocPitch(&data, &pitch, w*sizeof(T), h));
	}

	T* allocAndCopyToHost() {
		T* hostData = new T[w*h];
		cudaCheck(cudaMemcpy2D(hostData, w*sizeof(T), data, pitch, sizeof(T)*w, h, cudaMemcpyDeviceToHost));
		return hostData;
	}

	void copyToHost(T* dst) {
		cudaCheck(cudaMemcpy2D(dst, w*sizeof(T), data, pitch, sizeof(T)*w, h, cudaMemcpyDeviceToHost));
	}


	class reducer_buffer
	{
	public:
		Array2D<T> blockResults1;
		Array2D<T> blockResults2;
		T* hostBuf;

		enum {
			blockSize = 32,
			cpuThreshold = 32
		};

		reducer_buffer(size_t width, size_t height) {
			init(width,height);
		}

		reducer_buffer(const Array2D& t) {
			init(t.w, t.h);
		}

		void init(size_t width, size_t height) {
			int nBlockX = (width + blockSize - 1) / blockSize;
			blockResults1.init(nBlockX * height, 1);
			blockResults2.init( (nBlockX * height + blockSize - 1) / blockSize, 1);
			hostBuf = new T[cpuThreshold];
		}

		~reducer_buffer()
		{
			delete[] hostBuf;
		}
	};

	template<typename TBinOp, typename TPixelOp>
	T reduce(typename Array2D<T> & a, reducer_buffer& rbuf)
	{
		// Allocate memory for per-block results
		Array2D<T> *output = &rbuf.blockResults1, *input = &a;

		const int blockSize = reducer_buffer::blockSize;
		dim3 nThreads(blockSize, 1,1);

		// Reduce image
		int width = a.w;
		bool first=true;
		int sharedMemSize = (blockSize > 32) ? blockSize*sizeof(T) : 64*sizeof(T); // required by kernel for unrolled code
		int nBlockX = (width + blockSize - 1) / blockSize;
		dim3 nBlocks(nBlockX, input->h, 1);
		do {
			if (first) {
				reduceArray2D_k<T, TBinOp, blockSize, TPixelOp > <<<nBlocks, nThreads, sharedMemSize>>> (input->data, input->pitch/sizeof(T), output->data, width);
				first=false;
			} else {
				reduceArray2D_k<T, TBinOp, blockSize, pixel_value<T> > <<<nBlocks, nThreads, sharedMemSize>>> (input->data, input->pitch/sizeof(T), output->data, width);
			}
			width = nBlocks.x*nBlocks.y;
			nBlocks = dim3( (width + blockSize - 1)/blockSize, 1 ,1);
			std::swap(output, input);
			if (output == &a) output = &rbuf.blockResults2;
		} while (width > reducer_buffer::cpuThreshold);

		// Copy to host memory. Block results are now in 'input' due to last std::swap
	//	input->copyToHost(rbuf.hostBuf);
		T* results=input->allocAndCopyToHost();
		T resultValue = 0.0f;
		TBinOp binary_op;
		for (int x=0;x<width;x++) {
			resultValue = binary_op(resultValue, results[x]);
		}
		delete[] results;
		return resultValue;
	}

	T sum(reducer_buffer& r)
	{
		return reduce<thrust::plus<T>, pixel_value<T> >(*this, r);
	}
	T momentX(reducer_buffer& r)
	{
		return reduce<thrust::plus<T>, pixel_COM_x<T> >(*this, r);
	}
	T momentY(reducer_buffer& r)
	{
		return reduce<thrust::plus<T>, pixel_COM_y<T> >(*this, r);
	}

	void bindTexture(texture<T, cudaTextureType1D, cudaReadModeNormalizedFloat>& tex)
	{
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
		tex.addressMode[0] = cudaAddressModeClamp;
		tex.addressMode[1] = cudaAddressModeClamp;
		tex.filterMode = cudaFilterModePoint;
		tex.normalized = false; // no normalized texture coordinates

		size_t offset;
		cudaBindTexture2D(&offset, &tex, data, &channelDesc, w, h, pitch);
	}


};



