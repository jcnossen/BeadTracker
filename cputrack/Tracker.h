
// Defines the interface for trackers (both CPU and CUDA)

#pragma once

#include <vector>
#include "utils.h"

#ifdef TRK_USE_DOUBLE
	typedef double xcor_t;
#else
	typedef float xcor_t;
#endif



class TrackerImageBuffer
{
public:
	TrackerImageBuffer() {}

	virtual ~TrackerImageBuffer() {}
	virtual void Assign(ushort* data, int pitch) = 0;
};

class Tracker
{
protected:
	int width, height;

public:
	Tracker() {}
	virtual ~Tracker() {}

	int GetWidth() { return width; }
	int GetHeight() { return height; }

	virtual vector2f ComputeXCor(vector2f initial) = 0;
	virtual vector2f ComputeXCor2D() = 0;
	virtual vector2f ComputeXCorInterpolated(vector2f initial, int iterations) = 0;
	virtual void SetImage16Bit(ushort* srcImage, uint w, uint h, uint srcpitch) = 0;
	virtual void SetImage8Bit(uchar* srcImage, uint w, uint h, uint srcpitch) = 0;
	virtual void SetImageFloat(float* srcImage) = 0;

	virtual vector2f ComputeCOM(float median) = 0;
	virtual void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float radius, vector2f center) = 0;

	virtual void SetZLUT(float* data, int planes,int res, float profile_radius) = 0;
	virtual float ComputeZ(vector2f center, int angularSteps) = 0; // radialSteps is given by zlut_res. Returns normalized Z position
	virtual float ComputeMedian() = 0;
		
	// Debug stuff
	virtual float* GetDebugImage() { return 0; }
	virtual bool GetLastXCorProfiles(std::vector<xcor_t>& xprof, std::vector<xcor_t>& yprof, 
		std::vector<xcor_t>& xconv, std::vector<xcor_t>& yconv) { return false; }

	virtual void SelectImageBuffer(TrackerImageBuffer* b) = 0;
};

DLL_EXPORT Tracker* CreateTrackerInstance(int w,int h,int xcorw);
TrackerImageBuffer* CreateTrackerImageBuffer(int w,int h);
