/*
GPU Simulation and single-molecule localization for micromirror optics
Author: Jelmer Cnossen (jcnossen at gmail)
License: GPL
*/

#include "GPUBase.h"
#include "GPUImage.h"

#define NPP_CALL(STMT) {NppStatus _status = STMT; }
	//if( (_status) != NPP_NO_ERROR) throwError( "NPP error for call %s. Code: %d", #STMT, _status); }

void GPUImage::free()
{
	if (d_copyBuf) {
		cudaFree(d_copyBuf);
		d_copyBuf=0;
	}
	if (d_img) {
		cudaFree(d_img);
		d_img=0;
	}
	w=h=0;
}

void GPUImage::init(int w,int h)
{
	this->w=w;
	this->h=h;
	cudaMalloc(&d_img, w*h*sizeof(float));
	clear();
}

void GPUImage::clear()
{
	cudaMemset(d_img, 0, sizeof(float)*w*h);
}


float GPUImage::abssum(cudaStream_t s) const {
	if (s) cublasSetKernelStream(s);

	return cublasSasum(w*h, d_img, 1);
}

void GPUImage::copyTo(GPUImage& img) const {
	if (img.w!=w||img.h!=h) 
		img.resize(w,h);
	cudaMemcpy(img.d_img, d_img, w*h*sizeof(float), cudaMemcpyDeviceToDevice);
}

float GPUImage::normalizeSum(float wantedsum, cudaStream_t s)
{
	float sum=abssum(s);
	if (s) nppSetStream(s);
	NPP_CALL(nppsMulC_32f_I(wantedsum/sum, d_img, w*h));
	return sum;
}

void GPUImage::addfloat(float value, cudaStream_t s)
{
	if (s) nppSetStream(s);
	NPP_CALL(nppsAddC_32f_I(value, d_img, w*h));
}

float GPUImage::normalizeMax(float wantedmax, cudaStream_t s)
{
	float m;
	if (s) nppSetStream(s);

	int maxIndex = cublasIsamax(w*h, d_img, 1) - 1; // cublas uses fortran indexing (1-based)
	cudaMemcpy(&m, &d_img[maxIndex], sizeof(m), cudaMemcpyDeviceToHost);
	if (m > 0.0f) {
		NPP_CALL(nppsMulC_32f_I(wantedmax/m, d_img, w*h));
	}
	return m;
}



float GPUImage::normalizeMinMax(float wantedmin, float wantedmax, cudaStream_t s)
{
	float max,min;
	if (s) nppSetStream(s);
	float wanteddif = wantedmax-wantedmin;

	int maxIndex = cublasIsamax(w*h, d_img, 1) - 1; // cublas uses fortran indexing (1-based)
	int minIndex = cublasIsamin(w*h, d_img, 1) - 1;

	cudaMemcpy(&max, &d_img[maxIndex], sizeof(max), cudaMemcpyDeviceToHost);
	cudaMemcpy(&min, &d_img[minIndex], sizeof(min), cudaMemcpyDeviceToHost);
	float dif=max-min;
	if (dif < 0.0f) { 
		dif=max; max=min; min=dif; dif=max-min;
	}
	if (dif > 0.0f) {
		NPP_CALL(nppsMulC_32f_I(wanteddif/dif, d_img, w*h));
		NPP_CALL(nppsAddC_32f_I(wantedmin - min * wanteddif / dif, d_img, w*h));
	}
	return dif;
}


__global__ void ClampImage(float* d, float maxVal, int2 size)
{
	ITEMIDX();
	float v = d[idx];
	if (v > maxVal) v = maxVal;
	d[idx] = v;
}

void GPUImage::clamp(float max, cudaStream_t s) {
	int2 size;
	size.x = w; size.y = h;
	ClampImage<<<dim3(w/BLKSIZE, h/BLKSIZE), dim3(BLKSIZE,BLKSIZE),0,s>>>(d_img, max, size);
}

void GPUImage::resize(int w,int h)
{
	if (this->w != w || this->h != h) {
		free();
		init(w,h);
	}
}

bool GPUImage::haveNaNs()
{
	float val = cublasSasum(w*h,d_img,1);
	return val!=val;
}


void GPUImage::add(const GPUImage& src, cudaStream_t s)
{
	if (s) nppSetStream(s);
	NPP_CALL(nppsAdd_32f_I(src.d_img, d_img, w*h));
}


GPUImage* GPUImage::buildFrom8bitStrided(uint8_t* data, int pixelsPerLine, int w,int h)
{
	GPUImage* img = new GPUImage(w,h);

	return img;
}


