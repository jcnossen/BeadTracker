
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <stdio.h>
#include <windows.h>
#include <cstdarg>

#include "../cudatrack/utils.h"
#include "../cudatrack/Array2D.h"

#include <thrust/functional.h>

#include "../cudatrack/tracker.h"

#pragma pack(push,4)
typedef struct TestItem 
{
	float a,b;
};
#pragma pack(pop)

__global__ void sumArrayKernel(const float* src, float *blockResults, int N, TestItem x)
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
		TestItem ti;
		 ti.a = 12; ti.b = 123;
		 float2 Y = { 1,2};
		sumArrayKernel<<<dim3(numBlocks,1,1), dim3(blockSize, 1,1), blockSize*sizeof(float)>>>(d_data, d_out, curLength, ti);
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

static DWORD startTime = 0;
void BeginMeasure() { startTime = GetTickCount(); }
void EndMeasure(const std::string& msg) {
	DWORD endTime = GetTickCount();
	dbgout(SPrintf("%s: %d ms\n", msg.c_str(), (endTime-startTime)));
}

int main()
{
	testLinearArray();


	Tracker tracker(512,512);

	tracker.loadTestImage(256,256, 10);
	vector2f COM = tracker.ComputeCOM();
	vector2f xcor = tracker.XCorLocalize(COM);

	/*
	int W=2024, H=1024;
	int NumRuns = 1;

	float* d = new float[W*H];
	for (int y=0;y<H;y++) 
		for(int x=0;x<W;x++) 
			d[y*W+x] = rand() / (float)RAND_MAX * 10.0f;
	Array2D<float> a(W,H, d);

	float sum = 0.0;
	BeginMeasure();
	for (int run=0;run<NumRuns;run++) {
		float comX=0.0, comY=0.0;
		for (int y=0;y<H;y++) {
			for(int x=0;x<W;x++) 
			{
				float value=d[y*W+x];
				sum += value;
//				comX += v*x;
	//			comY += v*y;
			}
		}
	}
	EndMeasure("CPU sum");

	Array2D<float>::reducer_buffer rbuf(a);
	dbgout(SPrintf("GPU result: %f, CPU result: %f\n", a.sum(rbuf), sum));
*/
	return 0;
}
