#pragma once
#include "../cudatrack/utils.h"

#include "fftw/fftw3.h"
#include <complex>
#include <vector>

typedef uchar pixel_t;
typedef std::complex<float> complexf;

const int XCorProfileLen = 32;

class CPUTracker
{
public:
	int width, height;

	float *srcImage, *debugImage;
	complexf *fft_out, *fft_revout;
	fftwf_plan fft_plan_fw, fft_plan_bw;
	std::vector<vector2f> radialDirs;

	float* zlut; // zlut[plane * zlut_res + r]
	int zlut_planes, zlut_res;
	std::vector<float> rprof, rprof_diff;

	int xcorw;
	std::vector<float> X_xc, X_xcr, X_result;
	std::vector<float> Y_xc, Y_xcr, Y_result;


	float getPixel(int x, int y) { return srcImage[width*y+x]; }
	float Interpolate(float x,float y);
	CPUTracker(int w, int h);
	~CPUTracker();
	vector2f ComputeXCor(vector2f initial, int iterations);
	void XCorFFTHelper(float* xc, float* xcr, float* result);
	// Compute the interpolated index of the maximum value in the result array
	float ComputeMaxInterp(const std::vector<float>& v);
	template<typename TPixel>
	void SetImage(TPixel* srcImage, uint w, uint h, uint srcpitch);

	vector2f ComputeCOM(float median);
	void RemoveBackground(float median);
	void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float radius, vector2f center);

	void Normalize(float *image=0);
	void SetZLUT(float* data, int planes,int res);
	float ComputeZ(vector2f center, int angularSteps, float radius); // radialSteps is given by zlut_res
};


ushort* floatToNormalizedUShort(float *data, uint w,uint h);



template<typename TPixel>
void CPUTracker::SetImage(TPixel* data, uint w,uint h, uint pitchInBytes)
{
	uchar* bp = (uchar*)data;

	for (uint y=0;y<h;y++) {
		for (uint x=0;x<w;x++) {
			srcImage[y*w+x] = ((TPixel*)bp)[x];
		}
		bp += pitchInBytes;
	}
}


template<typename TPixel>
float ComputeMedian(TPixel* data, uint w, uint h, uint srcpitch, float* pMedian)
{
	float median;
	if (!pMedian || *pMedian<0.0f) {
		//TPixel* sortbuf = new TPixel[w*h/4];
		float total = 0.0f;
		// compute mean once per 4 rows to save time
		for (uint y=0;y<h/4;y++) {
			for (uint x=0;x<w;x++) {
				//sortbuf[y*w+x] = ((TPixel*)((uchar*)data + y*4*srcpitch)) [x]; 
				total += ((TPixel*)((uchar*)data + y*4*srcpitch)) [x];
			}
		}

		//std::sort(sortbuf, sortbuf+(w*h/4));
//		median = *pMedian = sortbuf[w*h/8];
		median = total / (w*h/4);
		if (pMedian) *pMedian=median;
		//delete[] sortbuf;
	} else
		median = *pMedian;

	return median;
}
