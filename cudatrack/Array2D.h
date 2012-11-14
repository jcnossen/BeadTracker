/*
CUDA Tracker Array2D.
Contains the CUDA code to apply per-pixel calculations to arrays and sum them by reduce.
*/
#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

#include "cuda_shared_mem.h"

#include "std_incl.h"

void throwCudaError(cudaError_t err);

inline void cudaCheck(cudaError_t err) {
	if (err != cudaSuccess) 
		throwCudaError(err);
}

#define SHAREDCODE __host__ __device__

namespace gpuArray {

template<typename T>
class pixel_value
{
public:
	SHAREDCODE T operator()(const T& value, const unsigned int x, const unsigned int y) {
		return value;
	}
};

// Center of mass X
template<typename T>
class pixel_COM_x
{
public:
	SHAREDCODE T operator()(const T& value, const unsigned int x, const unsigned int y) {
		return value * x;
	}
};

// Center of mass Y
template<typename T>
class pixel_COM_y
{
public:
	SHAREDCODE T operator()(const T& value, const unsigned int x, const unsigned int y) {
		return value * y;
	}
};

template<typename T, typename TFirstOp, typename TSecondOp>
class pixel_math_combiner
{
public:
	TFirstOp first;
	TSecondOp second;
	SHAREDCODE T operator()(const T& value, const unsigned int x, const unsigned int y) {
		return second(first(value, x, y), x, y);
	}
};

// Converts a binary functor to a pixel op
template<typename T, typename TBinaryOp>
class pixel_value_binary_adapter
{
public:
	T operand;
	SHAREDCODE T operator()(const T& value, const unsigned int x, const unsigned int y) { 
		TBinaryOp op;
		return op(value, operand); 
	}
};


template<typename T, typename TPixelOp>
__global__ void apply_pixel_op(T* data, uint pitch, uint width, uint height, TPixelOp pixel_op)
{
	uint x = threadIdx.x + blockIdx.x * blockDim.x;
	uint y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < width && y < height)
	{
		//TPixelOp pixel_op;
		data[y * pitch + x] = pixel_op(data[y * pitch + x], x, y);
	}
}

template<typename T, typename TCompute>
class Array2D;

template<typename T>
struct reducer_buffer;

template<typename T, typename TCompute, typename TBinOp, typename TPixelOp>
typename TCompute ReduceArray2D(Array2D<T, TCompute>& a, typename reducer_buffer<TCompute>& rbuf, TPixelOp pixel_op);


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

	Array2D(const Array2D& cp) {
		init(cp.w, cp.h);
		cudaCheck(cudaMemcpy2D(data, pitch, cp.data, cp.pitch, w*sizeof(T), h, cudaMemcpyDeviceToDevice));
	}

	Array2D& operator=(const Array2D& cp) {
		if (w != cp.w || h!=cp.h) {
			if (data) cudaFree(data);
			init(cp.w,cp.h);
		}
		cudaCheck(cudaMemcpy2D(data, pitch, cp.data, cp.pitch, w*sizeof(T), h, cudaMemcpyDeviceToDevice));
		return *this;
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

private:
	template<typename TReduceOp, typename TPixelOp>
	TCompute ReduceArrayHelper(reducer_buffer<TCompute>& r, TReduceOp reduce_op, TPixelOp pixel_op)
	{
		return ReduceArray2D<T, TCompute, TReduceOp, TPixelOp >(*this, r, pixel_op);
	}

public:

	TCompute sum(reducer_buffer<TCompute>& r)
	{
		return ReduceArrayHelper(r, thrust::plus<TCompute>(), pixel_value<TCompute>());
	}
	TCompute momentX(reducer_buffer<TCompute>& r)
	{
		return ReduceArray2D<T, TCompute, thrust::plus<TCompute>, pixel_COM_x<TCompute> >(*this, r, pixel_COM_x<TCompute>());
	}
	TCompute momentY(reducer_buffer<TCompute>& r)
	{
		return ReduceArray2D<T, TCompute, thrust::plus<TCompute>, pixel_COM_y<TCompute> >(*this, r, pixel_COM_y<TCompute>());
	}
	TCompute maximum(reducer_buffer<TCompute>& r)
	{
		return ReduceArray2D<T, TCompute, thrust::maximum<TCompute>, pixel_value<TCompute> >(*this, r, pixel_value<TCompute>());
	}
	TCompute minimum(reducer_buffer<TCompute>& r)
	{
		return ReduceArray2D<T, TCompute, thrust::minimum<TCompute>, pixel_value<TCompute> >(*this, r, pixel_value<TCompute>());
	}

	void normalize(reducer_buffer<TCompute>& r)
	{
		float maxValue = maximum(r);
		float minValue = minimum(r);
		multiplyAdd(1.0f / (maxValue-minValue), -minValue / (maxValue-minValue ));
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
	
	void unbindTexture(texture<T, cudaTextureType1D, cudaReadModeNormalizedFloat>& tex)
	{
		cudaUnbindTexture(&tex);
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
		/*
		T* d = allocAndCopyToHost();
		for (int y=0;y<h;y++)
			for (int x=0;x<w;x++)
				d[y*w+x] = pixel_op(d[y*w+x],x,y,pixel_op.data());
		set(d, sizeof(T)*w);
		delete[] d;*/
	}

	// does pixel = pixel * a + b
	void multiplyAdd(T a, T b)
	{
		pixel_value_binary_adapter<T, thrust::multiplies<T> > mul = {a};
		pixel_value_binary_adapter<T, thrust::plus<T> > add = {b};
		pixel_math_combiner< T,pixel_value_binary_adapter< T, thrust::multiplies<T> > , 
			pixel_value_binary_adapter<T, thrust::plus<T> > > combiner;
		combiner.first = mul; 
		combiner.second = add;
		applyPerPixel(combiner);
	}
	
	void copyTo(thrust::device_vector<T>& dst)
	{
		if(dst.size() != w*h) dst.resize(w*h);
		T* ptr = thrust::raw_pointer_cast(&dst[0]);
		cudaMemcpy2D(ptr, sizeof(T)*w, data, pitch, sizeof(T)*w, h, cudaMemcpyDeviceToDevice);
	}
};

template<typename T, typename TOut, typename TBinaryFunc, unsigned int blockSize, typename TPixelOp>
__global__ void reduceArray2D_k(const T* src, size_t spitch, TOut* out, size_t width, TPixelOp pixel_op)
{
	TOut* sdata = SharedMemory<TOut>();

	int tid = threadIdx.x;
	int xpos = threadIdx.x + blockIdx.x * blockSize;

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
	cudaStream_t stream;

	enum {
		blockSize = 64,
		cpuThreshold = 32
	};

	reducer_buffer(size_t width, size_t height) {
		init(width,height);
		cudaStreamCreate(&stream);
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
		cudaStreamDestroy(stream);
	}
};


template<typename T, typename TCompute, typename TBinOp, typename TPixelOp>
typename TCompute ReduceArray2D(Array2D<T, TCompute>& a, reducer_buffer<TCompute>& rbuf, TPixelOp pixel_op)
{
	// Allocate memory for per-block results
	Array2D<TCompute> *output = &rbuf.blockResults1, *input = &rbuf.blockResults2;

	const int blockSize = reducer_buffer<TCompute>::blockSize;
	dim3 nThreads(blockSize, 1,1);

	// Reduce image
	int width = a.w;
	int sharedMemSize = (blockSize > 32) ? blockSize*sizeof(TCompute) : 64*sizeof(TCompute); // required by kernel for unrolled code
	int nBlockX = (width + blockSize - 1) / blockSize;
	dim3 nBlocks(nBlockX, a.h, 1);
	reduceArray2D_k<T, TCompute, TBinOp, blockSize, TPixelOp > <<<nBlocks, nThreads, sharedMemSize, rbuf.stream>>> (a.data, a.pitch/sizeof(T), output->data, width, pixel_op);
	while (nBlocks.x * nBlocks.y > reducer_buffer<TCompute>::cpuThreshold) {
		width = nBlocks.x*nBlocks.y;
		nBlocks = dim3( (width + blockSize - 1)/blockSize, 1 ,1);
		std::swap(output, input);

		reduceArray2D_k<TCompute, TCompute, TBinOp, blockSize, pixel_value<TCompute> > <<<nBlocks, nThreads, sharedMemSize, rbuf.stream>>> 
			(input->data, input->pitch/sizeof(TCompute), output->data, width, pixel_value<TCompute>());
	} 

	// Copy to host memory. 
	output->copyToHost(rbuf.hostBuf);
	TCompute resultValue = 0.0f;
	TBinOp binary_op;
	for (int x=0;x<nBlocks.x;x++) {
		resultValue = binary_op(resultValue, rbuf.hostBuf[x]);
	}
	return resultValue;
}

} // end of gpuArray namespace
