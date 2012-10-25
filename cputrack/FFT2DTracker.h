
#pragma once

#include "scalar_types.h"

/*
2D FFT functionality is located in a seperate class, as the buffers are a lot bigger. This way, the memory is only allocated if ComputeXCor2D is actually used
*/
class FFT2DTracker {
public:
	FFT2DTracker(int w,int h);
	~FFT2DTracker();
	vector2f ComputeXCor(float* image);
	float* GetAutoConvResults() { return mirror2D; }

	int width,height;
	fftw_plan_t plan_fw2D, plan_bw2D;
	float *mirror2D;
	complexc *fft_buf, *fft_buf_mirrored;

	vector2f ComputeMax2DInterpolated(float *img);
};



