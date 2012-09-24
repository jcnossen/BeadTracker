/*
CUDA Tracker Array2D.
Contains the CUDA code to apply per-pixel calculations to arrays and sum them by reduce.
*/
#include <cuda_runtime.h>
#include <thrust/functional.h>

#include "cuda_shared_mem.h"

#include "utils.h"

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



template<typename T, typename TPixelOp>
__global__ void apply_pixel_op(T* data, uint pitch, uint width, uint height, TPixelOp pixel_op)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		data[y * pitch + x] = pixel_op(data[y * pitch + x], x, y);
	}
}

template<typename T, typename TCompute>
class Array2D;

template<typename T>
struct reducer_buffer;

template<typename T, typename TCompute, typename TBinOp, typename TPixelOp>
typename TCompute ReduceArray2D(Array2D<T, TCompute>& a, typename reducer_buffer<TCompute>& rbuf);


/* 
2D Array using CUDA pitched device memory 
T: Data storage 
TCompute: Compute type
*/
template<typename T, typename TCompute=T>
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
	
	void copyToHost(T* dst, uint dstPitch) {
		cudaCheck(cudaMemcpy2D(dst, dstPitch, data, pitch, sizeof(T)*w, h, cudaMemcpyDeviceToHost));
	}

	void set(T* src, uint srcPitch) {
		cudaCheck(cudaMemcpy2D(data, pitch, src, srcPitch, w*sizeof(T), h, cudaMemcpyHostToDevice));
	}


	TCompute sum(reducer_buffer<TCompute>& r)
	{
		return ReduceArray2D<T, TCompute, thrust::plus<TCompute>, pixel_value<TCompute> >(*this, r);
	}
	TCompute momentX(reducer_buffer<TCompute>& r)
	{
		return ReduceArray2D<T, TCompute, thrust::plus<TCompute>, pixel_COM_x<TCompute> >(*this, r);
	}
	TCompute momentY(reducer_buffer<TCompute>& r)
	{
		return ReduceArray2D<T, TCompute, thrust::plus<TCompute>, pixel_COM_y<TCompute> >(*this, r);
	}
	TCompute maximum(reducer_buffer<TCompute>& r)
	{
		return ReduceArray2D<T, TCompute, thrust::maximum<TCompute>, pixel_value<TCompute> >(*this, r);
	}
	TCompute minimum(reducer_buffer<TCompute>& r)
	{
		return ReduceArray2D<T, TCompute, thrust::minimum<TCompute>, pixel_value<TCompute> >(*this, r);
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

	dim3 compute_nBlocks(dim3 numThreads)
	{
		return dim3((w + numThreads.x - 1)/ numThreads.x , (h + numThreads.y - 1)/numThreads.y, 1);
	}

	template<typename TPixelOp>
	void applyPerPixel(TPixelOp pixel_op)
	{
		const uint BlockW=128;
		dim3 nThreads(BlockW, 1,1);
		dim3 nBlocks = compute_nBlocks(nThreads);
		apply_pixel_op<T, TPixelOp> <<<nThreads, nBlocks, 0>>> (data, pitch/sizeof(T), w, h, pixel_op);

		/*T* d = allocAndCopyToHost();
		for (int y=0;y<h;y++)
			for (int x=0;x<w;x++)
				d[y*w+x] = pixel_op(d[y*w+x],x,y);
		set(d, sizeof(T)*w);
		delete[] d;*/
	}


};

template<typename T, typename TOut, typename TBinaryFunc, unsigned int blockSize, typename TPixelOp>
__global__ void reduceArray2D_k(const T* src, size_t spitch, TOut* out, size_t width)
{
	TOut* sdata = SharedMemory<TOut>();

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
		volatile TOut *smem = sdata;
		if (blockSize >=  64) { smem[tid] = reduce_op((TOut)smem[tid], (TOut)smem[tid + 32]); }
		if (blockSize >=  32) { smem[tid] = reduce_op((TOut)smem[tid], (TOut)smem[tid + 16]); }
		if (blockSize >=  16) { smem[tid] = reduce_op((TOut)smem[tid], (TOut)smem[tid +  8]); }
		if (blockSize >=   8) { smem[tid] = reduce_op((TOut)smem[tid], (TOut)smem[tid +  4]); }
		if (blockSize >=   4) { smem[tid] = reduce_op((TOut)smem[tid], (TOut)smem[tid +  2]); }
		if (blockSize >=   2) { smem[tid] = reduce_op((TOut)smem[tid], (TOut)smem[tid +  1]); }
	}

	// write result 
	if (tid==0) out[blockIdx.y*gridDim.x + blockIdx.x] = sdata[0];
}


template<typename TCompute>
class reducer_buffer
{
public:
	Array2D<TCompute> blockResults1;
	Array2D<TCompute> blockResults2;
	TCompute* hostBuf;

	enum {
		blockSize = 32,
		cpuThreshold = 32
	};

	reducer_buffer(size_t width, size_t height) {
		init(width,height);
	}

	void init(size_t width, size_t height) {
		int nBlockX = (width + blockSize - 1) / blockSize;
		blockResults1.init(nBlockX * height, 1);
		blockResults2.init( (nBlockX * height + blockSize - 1) / blockSize, 1);
		hostBuf = new TCompute[cpuThreshold];
	}

	~reducer_buffer()
	{
		delete[] hostBuf;
	}
};


template<typename T, typename TCompute, typename TBinOp, typename TPixelOp>
typename TCompute ReduceArray2D(Array2D<T, TCompute>& a, reducer_buffer<TCompute>& rbuf)
{
	// Allocate memory for per-block results
	Array2D<TCompute> *output = &rbuf.blockResults1, *input = &rbuf.blockResults2;

	const int blockSize = reducer_buffer<TCompute>::blockSize;
	dim3 nThreads(blockSize, 1,1);

	// Reduce image
	int width = a.w;
	int sharedMemSize = (blockSize > 32) ? blockSize*sizeof(TCompute) : 64*sizeof(TCompute); // required by kernel for unrolled code
	int nBlockX = (width + blockSize - 1) / blockSize;
	dim3 nBlocks(nBlockX, input->h, 1);
	reduceArray2D_k<T, TCompute, TBinOp, blockSize, TPixelOp > <<<nBlocks, nThreads, sharedMemSize>>> (a.data, a.pitch/sizeof(T), output->data, width);
	while (nBlocks.x * nBlocks.y > reducer_buffer<TCompute>::cpuThreshold) {
		width = nBlocks.x*nBlocks.y;
		nBlocks = dim3( (width + blockSize - 1)/blockSize, 1 ,1);
		std::swap(output, input);

		reduceArray2D_k<TCompute, TCompute, TBinOp, blockSize, pixel_value<TCompute> > <<<nBlocks, nThreads, sharedMemSize>>> (input->data, input->pitch/sizeof(TCompute), output->data, width);
	} 

	// Copy to host memory. Block results are now in 'input' due to last std::swap
//	input->copyToHost(rbuf.hostBuf);
	TCompute* results=output->allocAndCopyToHost();
	TCompute resultValue = 0.0f;
	TBinOp binary_op;
	for (int x=0;x<width;x++) {
		resultValue = binary_op(resultValue, results[x]);
	}
	delete[] results;
	return resultValue;
}
