#pragma once
#include "../cudatrack/utils.h"

#include "fftw/fftw3.h"
#include <complex>
#include <vector>

typedef uchar pixel_t;
typedef std::complex<float> complexf;

const int XCorProfileLen = 16;

class CPUTracker
{
public:
	int width, height;

	float *srcImage;
	float getPixel(int x, int y) { return srcImage[width*y+x]; }
	float interpolate(float x,float y);

	complexf *fft_out, *fft_revout;
	fftwf_plan fft_plan_fw, fft_plan_bw;

	int xcorw;
	std::vector<float> X_xc, X_xcr, X_result;
	std::vector<float> Y_xc, Y_xcr, Y_result;

	CPUTracker(uint w,uint h);
	~CPUTracker();
	vector2f ComputeXCor(vector2f initial, int iterations);
	void XCorFFTHelper(float* xc, float* xcr, float* result);
	// Compute the interpolated index of the maximum value in the result array
	float ComputeMaxInterp(const std::vector<float>& v);
	void SetImage(float* srcImage);

	template<typename TPixel>
	void bgcorrect(TPixel* data, uint w, uint h, uint srcpitch, float* pMedian);

};
