
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "std_incl.h"
#include "utils.h"

#include <cassert>
#include <cstdlib>
#include <stdio.h>
#include <windows.h>
#include <cstdarg>
#include <valarray>

#include "cudafft/cudafft.h"
#include "random_distr.h"

#include <stdint.h>
#include "cudaImageList.h"
#include "QueuedCUDATracker.h"

#define LSQFIT_FUNC __device__ __host__
#include "LsqQuadraticFit.h"

double getPreciseTime()
{
	uint64_t freq, time;

	QueryPerformanceCounter((LARGE_INTEGER*)&time);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	return (double)time / (double)freq;
}

std::string getPath(const char *file)
{
	std::string s = file;
	int pos = s.length()-1;
	while (pos>0 && s[pos]!='\\' && s[pos]!= '/' )
		pos--;
	
	return s.substr(0, pos);
}


texture<float, cudaTextureType2D, cudaReadModeElementType> xcor1D_images(0, cudaFilterModeLinear);

inline __device__ float2 mul_conjugate(float2 a, float2 b)
{
	float2 r;
	r.x = a.x*b.x + a.y*b.y;
	r.y = a.y*b.x - a.x*b.y;
	return r;
}

template<typename T>
__device__ T max_(T a, T b) { return a>b ? a : b; }
template<typename T>
__device__ T min_(T a, T b) { return a<b ? a : b; }

template<typename T, int numPts>
__device__ T ComputeMaxInterp(T* data, int len)
{
	int iMax=0;
	T vMax=data[0];
	for (int k=1;k<len;k++) {
		if (data[k]>vMax) {
			vMax = data[k];
			iMax = k;
		}
	}
	T xs[numPts]; 
	int startPos = max_(iMax-numPts/2, 0);
	int endPos = min_(iMax+(numPts-numPts/2), len);
	int numpoints = endPos - startPos;


	if (numpoints<3) 
		return iMax;
	else {
		for(int i=startPos;i<endPos;i++)
			xs[i-startPos] = i-iMax;

		LsqSqQuadFit<T> qfit(numpoints, xs, &data[startPos]);
		//printf("iMax: %d. qfit: data[%d]=%f\n", iMax, startPos, data[startPos]);
		//for (int k=0;k<numpoints;k++) {
	//		printf("data[%d]=%f\n", startPos+k, data[startPos]);
		//}
		T interpMax = qfit.maxPos();

		if (fabs(qfit.a)<1e-9f)
			return (T)iMax;
		else
			return (T)iMax + interpMax;
	}
}

texture<float, cudaTextureType2D, cudaReadModeElementType> smpImgRef(0, cudaFilterModeLinear);


__global__ void runCudaFFT(cudafft<float>::cpx_type *src, cudafft<float>::cpx_type *dst, cudafft<float>::KernelParams kparams)
{
	kparams.makeShared();
	cudafft<float>::transform(src,dst, kparams);
}



void TestKernelFFT()
{
	int N=256;
	cudafft<float> fft(N, false);

	std::vector< cudafft<float>::cpx_type > data(N), result(N);
	for (int x=0;x<N;x++)
		data[x].x = 10*cos(x*0.1f-5);

	fft.host_transform(&data[0], &result[0]);
	for (int x=0;x<N;x++)
		dbgprintf("[%d] %f+%fi\n", x, result[x].x, result[x].y);

	// now put data in video mem
	cudafft<float>::cpx_type *src,*d_result;
	int memSize = sizeof(cudafft<float>::cpx_type)*N;
	cudaMalloc(&src, memSize);
	cudaMemcpy(src, &data[0], memSize, cudaMemcpyHostToDevice);
	cudaMalloc(&d_result, memSize);

	int sharedMemSize = fft.kparams_size;
	for (int k=0;k<100;k++) {
		runCudaFFT<<<dim3(1),dim3(1),sharedMemSize>>>(src,d_result, fft.kparams);
	}

	std::vector< cudafft<float>::cpx_type > result2(N);
	cudaMemcpy(&result2[0], d_result, memSize, cudaMemcpyDeviceToHost);

	for (int i=0;i<N;i++) {
		cudafft<float>::cpx_type d=result2[i]-result[i];
		dbgprintf("[%d] %f+%fi\n", i, d.x, d.y);
	}

	cudaFree(src);
	cudaFree(d_result);
}

int main(int argc, char *argv[])
{
//	testLinearArray();

	std::string path = getPath(argv[0]);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	TestLocalizationSpeed();
//	TestKernelFFT();

	return 0;
}
