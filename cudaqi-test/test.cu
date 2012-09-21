
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <stdio.h>
#include <windows.h>
#include <cstdarg>

#include "../cudaqi/utils.h"
#include "../cudaqi/Array2D.h"

#include <thrust/functional.h>

void dbgout(std::string s);
std::string SPrintf(const char *fmt, ...);

void throwCudaError(cudaError_t err)
{
	std::string msg = SPrintf("CUDA error: %s", cudaGetErrorString(err));
	dbgout(msg);
	throw std::runtime_error(msg);
}



__global__ void sumArrayKernel(const float* src, float *blockResults, int N)
{
	float* sdata = SharedMemory<float>();

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

	int W=10, H=30;

	double* d = new double[W*H];
	double sum = 0.0;
	double comX=0.0, comY=0.0;
	for (int y=0;y<H;y++) {
		for(int x=0;x<W;x++) 
		{
			double v = d[y*W+x] = rand() / (float)RAND_MAX * 10.0f;
			sum += v;
			comX += v*x;
			comY += v*y;
		}
	}
	comX /= sum;
	comY /= sum;

	Array2D<double> a(W,H, d);
	
	double result = a.sum();
	dbgout(SPrintf("GPU result: %f, CPU result: %f\n", result, sum ));
	dbgout(SPrintf("COMX: GPU: %f, CPU: %f\n", a.momentX()/result, comX ));
	dbgout(SPrintf("COMY: GPU: %f, CPU: %f\n", a.momentY()/result, comY ));

		
	return 0;
}
