#pragma once

#include "Tracker.h"
#include "utils.h"

#include "scalar_types.h"


typedef uchar pixel_t;

class FFT2DTracker;

class CPUTrackerImageBuffer : public TrackerImageBuffer
{
public:
	void Assign(ushort* src_data, int pitch);
	~CPUTrackerImageBuffer ();

	int w,h;
	ushort* data;
};


class CPUTracker : public Tracker
{
public:
	int xcorProfileWidth;
	FFT2DTracker* tracker2D;

	float *srcImage, *debugImage;
	complexc *fft_out, *fft_revout;
	fftw_plan_t fft_plan_fw, fft_plan_bw;
	std::vector<vector2f> radialDirs;

	// The ZLUT system stores 'zlut_count' number of 2D zlut's, so every bead can be tracked with its own unique ZLUT.
	float* zluts; // size: zlut_planes*zlut_count*zlut_res,		indexing: zlut[index * (zlut_planes * zlut_res) + plane * zlut_res + r]
	int zlut_planes, zlut_res, zlut_count; 
	std::vector<float> rprof, rprof_diff;
	float zprofile_radius;
	
	int xcorw;
	std::vector<xcor_t> shiftedResult;
	std::vector<xcor_t> X_xc, X_xcr, X_result;
	std::vector<xcor_t> Y_xc, Y_xcr, Y_result;

	float& getPixel(int x, int y) { return srcImage[width*y+x]; }
	float Interpolate(float x,float y);
	CPUTracker(int w, int h, int xcorwindow=128, int xcorProfileWidth=32);
	~CPUTracker();
	void setXCorWindow(int xcorwindow);

	vector2f ComputeXCor(vector2f initial);
	vector2f ComputeXCor2D();
	vector2f ComputeXCorInterpolated(vector2f initial, int iterations);
	void XCorFFTHelper(xcor_t* xc, xcor_t* xcr, xcor_t* result);
	template<typename TPixel>
	void SetImage(TPixel* srcImage, uint w, uint h, uint srcpitch);
	void SetImage16Bit(ushort* srcImage, uint w, uint h, uint srcpitch) { SetImage(srcImage, w, h, srcpitch); }
	void SetImage8Bit(uchar* srcImage, uint w, uint h, uint srcpitch) { SetImage(srcImage, w, h, srcpitch); }
	void SetImageFloat(float* srcImage);

	vector2f ComputeCOM(float median);
	void RemoveBackground(float median);
	void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float radius, vector2f center);

	void Normalize(float *image=0);
	void SetZLUT(float* data, int planes, int res, int num_zluts, float prof_radius);
	float ComputeZ(vector2f center, int angularSteps, int zlutIndex); // radialSteps is given by zlut_res
	float ComputeMedian();

	bool GetLastXCorProfiles(std::vector<xcor_t>& xprof, std::vector<xcor_t>& yprof, 
		std::vector<xcor_t>& xconv, std::vector<xcor_t>& yconv);

	// Compute the interpolated index of the maximum value in the result array
	template<typename T> T ComputeMaxInterp(T* data, int len, int numpoints=5);

	void OutputDebugInfo();
	float* GetDebugImage() { return debugImage; }
	void SelectImageBuffer(TrackerImageBuffer* b);

};

void GenerateTestImage(float* data, int w, float xp, float yp, float size, float MaxPhotons);
CPUTracker* CreateCPUTrackerInstance(int w,int h,int xcorw);

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


template<typename T>
T CPUTracker::ComputeMaxInterp(T* data, int len, int numpoints)
{
	int iMax=0;
	T vMax=data[0];
	for (int k=1;k<len;k++) {
		if (data[k]>vMax) {
			vMax = data[k];
			iMax = k;
		}
	}
	int startPos = std::max(iMax-numpoints/2, 0);
	int endPos = std::min(iMax+(numpoints-numpoints/2), len);
	numpoints = endPos - startPos;

	if (numpoints<3) 
		return iMax;
	else {
		T *xs = (T*)ALLOCA(sizeof(T)*numpoints);
		for(int i=startPos;i<endPos;i++)
			xs[i-startPos] = i-iMax;
		LsqSqQuadFit<T> qfit(numpoints, xs, &data[startPos]);
		T interpMax = qfit.maxPos();

		return (T)iMax + interpMax;
	}
}


