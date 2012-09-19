
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdlib>
#include <stdio.h>
#include <windows.h>
#include <cstdarg>

#include "../cudaqi/utils.h"

void dbgout(std::string s);

__device__ void sumArrayKernel(float* data, float *tmp, int N)
{
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	float* src=data;
	float* dst=data;

	for (unsigned int k=1;k<N;k*=2) {
		tmp[
	}
}

float sumArray(float* data, int length)
{
	float *d_data;
	cudaMalloc(&d_data, length*4);
	cudaMemcpy(d_data, data, length*4, cudaMemcpyHostToDevice);

//	sumArrayKernel<<<
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


int main()
{
	int N = 100;
	float *d = new float[N];

	for (int x=0;x<N;x++)
		d[x] = rand() / (float)RAND_MAX * 10.0f;
	
	float result = sumArray(d, N);

	printf("result: %f\n", result);
	

	return 0;
}
