#pragma once
#include "../cudatrack/utils.h"

#include "fftw/fftw3.h"
#include <complex>
#include <vector>

#ifdef TRK_USE_DOUBLE
	typedef double xcor_t;
	typedef fftw_plan fftw_plan_t;
#else
	typedef float xcor_t;
	typedef fftwf_plan fftw_plan_t;
#endif

typedef uchar pixel_t;
typedef std::complex<xcor_t> complexc;

class CPUTracker
{
public:
	int width, height;
	int xcorProfileWidth;

	float *srcImage, *debugImage;
	complexc *fft_out, *fft_revout;
	fftw_plan_t fft_plan_fw, fft_plan_bw;
	std::vector<vector2f> radialDirs;

	//----------- 2D Cross-Correlation variables
	fftw_plan_t plan_fw2D, plan_bw2D;
	complexc* fft_out2D;

	float* zlut; // zlut[plane * zlut_res + r]
	int zlut_planes, zlut_res;
	std::vector<float> rprof, rprof_diff;

	int xcorw;
	std::vector<xcor_t> shiftedResult;
	std::vector<xcor_t> X_xc, X_xcr, X_result;
	std::vector<xcor_t> Y_xc, Y_xcr, Y_result;

	float& getPixel(int x, int y) { return srcImage[width*y+x]; }
	float Interpolate(float x,float y);
	CPUTracker(int w, int h, int xcorwindow=128);
	~CPUTracker();
	void setXCorWindow(int xcorwindow);

	vector2f Compute2DXCor();
	vector2f ComputeXCor(vector2f initial);
	vector2f ComputeXCorInterpolated(vector2f initial, int iterations);
	void XCorFFTHelper(xcor_t* xc, xcor_t* xcr, xcor_t* result);
	template<typename TPixel>
	void SetImage(TPixel* srcImage, uint w, uint h, uint srcpitch);

	vector2f ComputeCOM(float median);
	void RemoveBackground(float median);
	void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float radius, vector2f center);

	void Normalize(float *image=0);
	void SetZLUT(float* data, int planes,int res);
	float ComputeZ(vector2f center, int angularSteps, float radius); // radialSteps is given by zlut_res
	float ComputeMedian();

	// Compute the interpolated index of the maximum value in the result array
	template<typename T> T ComputeMaxInterp(const std::vector<T>& r);

	void OutputDebugInfo();
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

template<typename TPixel>
void normalize(TPixel* d, uint w,uint h)
{
	TPixel maxv = d[0];
	TPixel minv = d[0];
	for (uint k=0;k<w*h;k++) {
		maxv = max(maxv, d[k]);
		minv = min(minv, d[k]);
	}
	for (uint k=0;k<w*h;k++)
		d[k]=(d[k]-minv)/(maxv-minv);
}


template<typename T>
T CPUTracker::ComputeMaxInterp(const std::vector<T>& r)
{
	uint iMax=0;
	T vMax=0;
	for (uint k=0;k<r.size();k++) {
		if (r[k]>vMax) {
			vMax = r[k];
			iMax = k;
		}
	}
	if (iMax<2 || iMax>=r.size()-2)
		return iMax; // on the edge, so we ignore the interpolation
	
	T xs[] = {-2, -1, 0, 1, 2};
	LsqSqQuadFit<T> qfit(5, xs, &r[iMax-2]);
	xcor_t interpMax = qfit.maxPos();

	return (T)iMax + interpMax;
}

