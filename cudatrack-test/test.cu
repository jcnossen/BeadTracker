
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

#include "nivision.h" // write PNG file

/*
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

	//printf(" Hi %d " , i);
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


void testLinearArray()
{
	int N = 10;
	float *d = new float[N];
	
	for (int x=0;x<N;x++)
		d[x] = rand() / (float)RAND_MAX * 10.0f;
	
	float result = sumHostArray(d, N);
	float cpuResult = 0.0f;
	for (int x=0;x<N;x++)
		cpuResult += d[x];
	dbgout(SPrintf("GPU result: %f. CPU result: %f\n", result, cpuResult));
}*/

static DWORD startTime = 0;
void BeginMeasure() { startTime = GetTickCount(); }
void EndMeasure(const std::string& msg) {
	DWORD endTime = GetTickCount();
	dbgout(SPrintf("%s: %d ms\n", msg.c_str(), (endTime-startTime)));
}

void saveImage(Array2D<float>& img, const char* filename)
{
	pixel_t *data = new pixel_t[img.w * img.h];
	img.copyToHost(data);

	pixel_t maxv = data[0];
	pixel_t minv = data[0];
	for (int k=0;k<img.w*img.h;k++) {
		maxv = max(maxv, data[k]);
		minv = min(minv, data[k]);
	}
	ushort *norm = new ushort[img.w*img.h];
	for (int k=0;k<img.w*img.h;k++)
		norm[k] = ((1<<16)-1) * (data[k]-minv) / (maxv-minv);

	Image* dst = imaqCreateImage(IMAQ_IMAGE_U16, 0);
	imaqSetImageSize(dst, img.w, img.h);
	imaqArrayToImage(dst, norm, img.w, img.h);
	delete[] data;
	delete[] norm;

	ImageInfo info;
	imaqGetImageInfo(dst, &info);
	int success = imaqWriteFile(dst, filename, 0);
	if (!success) {
		char *errStr = imaqGetErrorText(imaqGetLastError());
		std::string msg = SPrintf("IMAQ WriteFile error: %s\n", errStr);
		imaqDispose(errStr);
		dbgout(msg);
	}
	imaqDispose(dst);
}

std::string getPath(const char *file)
{
	std::string s = file;
	int pos = s.length()-1;
	while (pos>0 && s[pos]!='\\' && s[pos]!= '/' )
		pos--;
	
	return s.substr(0, pos);
}

int main(int argc, char *argv[])
{
//	testLinearArray();

	std::string path = getPath(argv[0]);

	Tracker tracker(256,256);

	tracker.loadTestImage(128,128, 30);

	Array2D<float> tmp(10,10);
	float tmpdata[100];
	float sumCPU=0.0f;
	for (int k=0;k<100;k++) {
		tmpdata[k]=k;
		sumCPU =tmpdata[k]+sumCPU;
	}
	tmp.set(tmpdata, sizeof(float)*10);
	reducer_buffer<float> rbuf(10,10);
	float sumGPU = tmp.sum(rbuf);

	dbgout(SPrintf("SumCPU: %f, SUMGPU: %f\n", sumCPU, sumGPU));
	
	Array2D<pixel_t, float>* data = (Array2D<pixel_t, float>*)tracker.getCurrentBufferImage();
	saveImage(*data, (path + "\\testImg.png").c_str());
	
	vector2f COM = tracker.ComputeCOM();
	dbgout(SPrintf("COM: %f, %f\n", COM.x,COM.y));
//	vector2f xcor = tracker.XCorLocalize(COM);


	return 0;
}
