
// Defines the interface for trackers (both CPU and CUDA)

#pragma once

#include <vector>
#include "utils.h"

class TrackerImageBuffer
{
public:
	TrackerImageBuffer() {}

	virtual ~TrackerImageBuffer() {}
	virtual void Assign(uchar* data, int pitch) = 0;
	virtual void Assign(ushort* data, int pitch) = 0;
	virtual void Assign(float* data, int pitch) = 0;
};

enum Localize2DType {
	// Flags for selecting 2D localization type
	LocalizeXCor1D = 0, // COM+XCor1D
	LocalizeOnlyCOM = 1, // use only COM
	LocalizeQI = 2, // COM+QI
	LocalizeXCor2D = 3,   // XCor2D
	Force32Bit = 0xffffffff
};

enum QTRK_PixelDataType
{
	QTrkU8 = 0,
	QTrkU16 = 1,
	QTrkFloat = 2
};


#pragma pack(push, 4)
// DONT CHANGE, Mapped to labview clusters!
struct LocalizationResult {
	uint zlutIndex;
	uint id;
	vector2f pos, firstGuess;
	uint locType;
	float z;
};
// DONT CHANGE, Mapped to labview clusters (QTrkSettings.ctl)!
struct QTrkSettings {
	int width, height;
	int numThreads;

	int xcorw;
	int profileWidth;
	int xcor1D_iterations;

	float zlut_minradius, zlut_maxradius;
	int zlut_angularsteps;

	int qi_iterations;
	int qi_radialsteps, qi_angularsteps;
	float qi_minradius, qi_maxradius;
};
#pragma pack(pop)

class QueuedTracker
{
protected:
	int width, height;
public:
	QueuedTracker() {}
	virtual ~QueuedTracker() {}

	int GetWidth() { return width; }
	int GetHeight() { return height; }

	virtual void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, Localize2DType locType, bool computeZ, uint id, uint zlutIndex=0) = 0;
	virtual int PollFinished(LocalizationResult* results, int maxResults) = 0;

	virtual void SetZLUT(float* data, int planes, int res, int numLUTs) = 0;
	virtual void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float radius, vector2f center) = 0;

	// Debug stuff
	virtual float* GetDebugImage() { return 0; }
};

