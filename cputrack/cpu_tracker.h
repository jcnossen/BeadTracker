#pragma once

#include "QueuedTracker.h"
#include "utils.h"
#include "scalar_types.h"
#include "kissfft.h"


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


class CPUTracker
{
public:
	int width, height;

	float *srcImage, *debugImage;
	complexc *fft_out, *fft_revout;
	std::vector<vector2f> radialDirs; // full circle for ZLUT
	std::vector<vector2f> quadrantDirs; // single quadrant

	// The ZLUT system stores 'zlut_count' number of 2D zlut's, so every bead can be tracked with its own unique ZLUT.
	float* zluts; // size: zlut_planes*zlut_count*zlut_res,		indexing: zlut[index * (zlut_planes * zlut_res) + plane * zlut_res + r]
	bool zlut_memoryOwner; // is this instance the owner of the zluts memory, or is it external?
	int zlut_planes, zlut_res, zlut_count, zlut_angularSteps; 
	std::vector<float> rprof, rprof_diff;
	float zprofile_radius;
	
	kissfft<xcor_t> fft_forward, fft_backward;
	int xcorw;
	std::vector<complexc> shiftedResult;
	std::vector<complexc> X_xc, X_xcr;
	std::vector<xcor_t> X_result;
	std::vector<complexc> Y_xc, Y_xcr;
	std::vector<xcor_t> Y_result;

	float& getPixel(int x, int y) { return srcImage[width*y+x]; }
	int GetWidth() { return width; }
	int GetHeight() { return height; }
	float Interpolate(float x,float y);
	CPUTracker(int w, int h, int xcorwindow=128);
	~CPUTracker();

	vector2f ComputeXCor(vector2f initial, int profileWidth=32);
	vector2f ComputeXCor2D();
	vector2f ComputeXCorInterpolated(vector2f initial, int iterations, int profileWidth=32);
	vector2f ComputeQI(vector2f initial, int iterations, int radialSteps, int angularStepsPerQuadrant, float radius);
	void XCorFFTHelper(complexc* xc, complexc* xcr, xcor_t* result);

	template<typename TPixel> void SetImage(TPixel* srcImage, uint srcpitch);
	void SetImage16Bit(ushort* srcImage, uint srcpitch) { SetImage(srcImage, srcpitch); }
	void SetImage8Bit(uchar* srcImage, uint srcpitch) { SetImage(srcImage, srcpitch); }
	void SetImageFloat(float* srcImage);

	vector2f ComputeBgCorrectedCOM();
	void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float radius, vector2f center);

	void Normalize(float *image=0);
	void SetZLUT(float* data, int planes, int res, int num_zluts, float prof_radius, int angularSteps, bool copyMemory);
	float ComputeZ(vector2f center, int angularSteps, int zlutIndex); // radialSteps is given by zlut_res

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
void CPUTracker::SetImage(TPixel* data, uint pitchInBytes)
{
	uchar* bp = (uchar*)data;

	for (int y=0;y<height;y++) {
		for (int x=0;x<width;x++) {
			srcImage[y*width+x] = ((TPixel*)bp)[x];
		}
		bp += pitchInBytes;
	}
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

