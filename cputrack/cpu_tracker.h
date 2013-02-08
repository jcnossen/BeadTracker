#pragma once

#include "QueuedTracker.h"
#include "utils.h"
#include "scalar_types.h"
#include "kissfft.h"


typedef uchar pixel_t;


class XCor1DBuffer {
public:
	XCor1DBuffer(int xcorw);
	~XCor1DBuffer();

	kissfft<xcor_t> fft_forward, fft_backward;
	int xcorw;
	std::vector<complexc> shiftedResult;
	std::vector<complexc> X_xc, X_xcr;
	std::vector<xcor_t> X_result;
	std::vector<complexc> Y_xc, Y_xcr;
	std::vector<xcor_t> Y_result;
	complexc *fft_out, *fft_revout;

	void XCorFFTHelper(complexc* xc, complexc* xcr, xcor_t* result);
	void OutputDebugInfo();
};

class CPUTracker
{
public:
	int width, height, xcorw;

	float *srcImage, *debugImage;
	float mean; // Updated by ComputeBgCorrectedCOM()
#ifdef _DEBUG
	float maxImageValue;
#endif
	std::vector<vector2f> radialDirs; // full circle for ZLUT

	// The ZLUT system stores 'zlut_count' number of 2D zlut's, so every bead can be tracked with its own unique ZLUT.
	float* zluts; // size: zlut_planes*zlut_count*zlut_res,		indexing: zlut[index * (zlut_planes * zlut_res) + plane * zlut_res + r]
	bool zlut_memoryOwner; // is this instance the owner of the zluts memory, or is it external?
	int zlut_planes, zlut_res, zlut_count, zlut_angularSteps; 
	float zlut_minradius, zlut_maxradius;
	bool zlut_useCorrelation;
	std::vector<float> zlut_radialweights;

	float* getZLUT(int index)  { return &zluts[zlut_res*zlut_planes*index]; }

	XCor1DBuffer* xcorBuffer;
	
	std::vector<vector2f> quadrantDirs; // single quadrant
	int qi_radialsteps;
	typedef float qi_t;
	typedef std::complex<qi_t> qic_t;
	kissfft<qi_t> *qi_fft_forward, *qi_fft_backward;

	float& getPixel(int x, int y) { return srcImage[width*y+x]; }
	int GetWidth() { return width; }
	int GetHeight() { return height; }
	CPUTracker(int w, int h, int xcorwindow=128);
	~CPUTracker();
	bool KeepInsideBoundaries(vector2f *center, float radius);
	bool CheckBoundaries(vector2f center, float radius);
	vector2f ComputeXCorInterpolated(vector2f initial, int iterations, int profileWidth, bool& boundaryHit);
	vector2f ComputeQI(vector2f initial, int iterations, int radialSteps, int angularStepsPerQuadrant, float minRadius, float maxRadius, bool& boundaryHit);

	qi_t QI_ComputeOffset(qic_t* qi_profile, int nr);
	float ComputeAsymmetry(vector2f center, int radialSteps, int angularSteps, float minRadius, float maxRadius, float *dstAngProf=0);

	template<typename TPixel> void SetImage(TPixel* srcImage, uint srcpitch);
	void SetImage16Bit(ushort* srcImage, uint srcpitch) { SetImage(srcImage, srcpitch); }
	void SetImage8Bit(uchar* srcImage, uint srcpitch) { SetImage(srcImage, srcpitch); }
	void SetImageFloat(float* srcImage);

	vector2f ComputeBgCorrectedCOM();
	void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius, vector2f center, bool crp, bool* boundaryHit=0);
	void ComputeQuadrantProfile(qi_t* dst, int radialSteps, int angularSteps, int quadrant, float minRadius, float maxRadius, vector2f center);

	void Normalize(float *image=0);
	void SetZLUT(float* data, int planes, int res, int num_zluts, float minradius, float maxradius, int angularSteps, bool copyMemory, bool useCorrelation, float* radialweights=0);
	float ComputeZ(vector2f center, int angularSteps, int zlutIndex, bool crp, bool* boundaryHit=0, float* profile=0, float* cmpprof=0 ); // radialSteps is given by zlut_res

	bool GetLastXCorProfiles(std::vector<xcor_t>& xprof, std::vector<xcor_t>& yprof, 
		std::vector<xcor_t>& xconv, std::vector<xcor_t>& yconv);

	void OutputDebugInfo();
	float* GetDebugImage() { return debugImage; }
};


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

	mean=0.0f;
}



template<typename T>
T ComputeMaxInterp(T* data, int len, int numpoints=7)
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
		double *xs = (double*)ALLOCA(sizeof(double)*numpoints);
		double *ys = (double*)ALLOCA(sizeof(double)*numpoints);
		for(int i=startPos;i<endPos;i++) {
			xs[i-startPos] = i-iMax;
			ys[i-startPos] = data[i];
		}
		LsqSqQuadFit<double> qfit(numpoints, xs, ys);
		if (fabs(qfit.a)<1e-9)
			return (T)iMax;
		else
			return (T)(iMax + qfit.maxPos());
	}
}

