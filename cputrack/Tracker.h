
// Defines the interface for trackers (both CPU and CUDA)

#pragma once

#include <vector>
#include "../cudatrack/utils.h"

#ifdef TRK_USE_DOUBLE
	typedef double xcor_t;
#else
	typedef float xcor_t;
#endif



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
	virtual vector2f ComputeXCorInterpolated(vector2f initial, int iterations) = 0;
	virtual void SetImage16Bit(ushort* srcImage, uint w, uint h, uint srcpitch) = 0;
	virtual void SetImage8Bit(uchar* srcImage, uint w, uint h, uint srcpitch) = 0;

	virtual vector2f ComputeCOM(float median) = 0;
	virtual void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float radius, vector2f center) = 0;

	virtual void SetZLUT(float* data, int planes,int res) = 0;
	virtual float ComputeZ(vector2f center, int angularSteps, float radius) = 0; // radialSteps is given by zlut_res
	virtual float ComputeMedian() = 0;
		
	// Debug stuff
	virtual float* GetDebugImage() { return 0; }
	virtual bool GetLastXCorProfiles(std::vector<xcor_t>& xprof, std::vector<xcor_t>& yprof, 
		std::vector<xcor_t>& xconv, std::vector<xcor_t>& yconv) { return false; }
};
