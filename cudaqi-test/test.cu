
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <stdio.h>
#include <windows.h>
#include <cstdarg>

#include "../cudaqi/utils.h"
#include "../cudaqi/Array2D.h"

void dbgout(std::string s);


extern __shared__ float sdata[];

__global__ void sumArrayKernel(const float* src, float *blockResults, int N)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	sdata[threadIdx.x] = i<N ? src[i] : 0;
	unsigned int tid = threadIdx.x;

	__syncthreads();

	for (unsigned int k=blockDim.x/2;k>=1;k>>=1) {
		if(tid < k) {
			sdata[tid] = sdata[tid] + sdata[tid+k];
		}
		__syncthreads();
	}

	blockResults[blockIdx.x] = sdata[0];
}

float sumDeviceArray(float *d_data, int length)
{
	const int blockSize = 256;
	const int cpuThreshold = 128;

	int numBlocks = (length + blockSize - 1) / blockSize;
	float* d_blockResults, *d_out;
	cudaMalloc(&d_blockResults, sizeof(float)*numBlocks);
	int curLength = length;

	d_out = d_blockResults;
	while (curLength > cpuThreshold) {
		numBlocks = (curLength + blockSize - 1) / blockSize;
		sumArrayKernel<<<dim3(numBlocks,1,1), dim3(blockSize, 1,1), blockSize*sizeof(float)>>>(d_data, d_out, curLength);
		// now reduced to 
		curLength = numBlocks;
		std::swap(d_data, d_out);
	}

	float resultValues[cpuThreshold];
	cudaMemcpy(resultValues, d_data, sizeof(float)*curLength, cudaMemcpyDeviceToHost);
	cudaFree(d_blockResults);

	float result = 0.0f;
	for (int x=0;x<curLength;x++)
		result += resultValues[x];
	return result;
}

float sumHostArray(float* data, int length)
{
	float *d_data;
	cudaMalloc(&d_data, length*4);
	cudaMemcpy(d_data, data, length*4, cudaMemcpyHostToDevice);

	float result = sumDeviceArray(d_data, length);	

	cudaFree(d_data);
	return result;
}


__global__ void reduceArray2D(const float* src, size_t spitch, float* out, size_t dpitch, size_t width, size_t height)
{

}

template<int BlockSize>
dim3 computeBlocksize(size_t w,size_t h) {
	return dim3 ( (a.w + BlockSize - 1) / BlockSize, (a.h + BlockSize - 1) / BlockSize , 1);
}

float sumArray2D(Array2D<float> & a)
{
	const int blockSize = 32;
	const int cpuThreshold = 256;
	// Allocate memory for per-block results
	dim3 nBlocks = computeBlocksize<blockSize>(a.w,a.h);
	Array2D<float> blockResults(nBlocks.x, nBlocks.y);

	dim3 sz(a.w, a.h, 1);
	while (sz.x * sz.y > cpuThreshold) {
		nBlocks = computeBlocksize<blockSize>(sz.x, sz.y);
		reduceArray2D<<<nBlocks, dim3(blockSize, blockSize,1), sizeof(float)*blockSize*blockSize>>> (a.data, a.pitch, blockResults.data, blockResults.pitch, sz.x, sz.y);
		sz = nBlocks;
	}
}

std::string SPrintf(const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);

	char buf[512];
	VSNPRINTF(buf, sizeof(buf), fmt, ap);

	va_end(ap);
	return buf;
}

void dbgout(std::string s) {
	OutputDebugString(s.c_str());
}


void testLinearArray()
{
	int N = 78233;
	float *d = new float[N];
	
	for (int x=0;x<N;x++)
		d[x] = rand() / (float)RAND_MAX * 10.0f;
	
	float result = sumHostArray(d, N);
	float cpuResult = 0.0f;
	for (int x=0;x<N;x++)
		cpuResult += d[x];
	dbgout(SPrintf("GPU result: %f. CPU result: %f\n", result, cpuResult));
}

int main()
{

//	testLinearArray();

	int W=256, H=256;

	float* d = new float[W*H];
	double sum = 0.0;
	for(int x=0;x<W*H;x++) {
		d[x] = x%1024;
		sum += (double)d[x];
	}

	Array2D<float> a(W,H, d);
	float result = sumArray2D(a);

	dbgout(SPrintf("GPU result: %f, CPU result: %f\n", result, sum ));
		
	return 0;
}
