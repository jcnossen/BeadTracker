
// Defines the interface for trackers (both CPU and CUDA)

#pragma once

#include "std_incl.h" 

enum LocalizeType {
	// Flags for selecting 2D localization type
	LocalizeOnlyCOM = 0, // use only COM
	LocalizeXCor1D = 1, // COM+XCor1D
	LocalizeQI = 2, // COM+QI
	LocalizeXCor2D = 3,   // XCor2D

	Localize2DMask = 7,
	LocalizeZ = 16,
	LocalizeBuildZLUT = 32,
	Force32Bit = 0xffffffff
};

enum QTRK_PixelDataType
{
	QTrkU8 = 0,
	QTrkU16 = 1,
	QTrkFloat = 2
};


#pragma pack(push, 1)
// DONT CHANGE, Mapped to labview clusters!
struct LocalizationResult {
	uint id;
	int zlutIndex;
	uint locType;
	vector2f pos;
	float z;
	vector2f firstGuess;
	uint error;
};
// DONT CHANGE, Mapped to labview clusters (QTrkSettings.ctl)!
struct QTrkSettings {
	QTrkSettings() {
		width = height = 150;
		numThreads = -1;
		maxQueueSize = 200;
		xc1_profileLength = 128; 
		xc1_profileWidth = 32;
		xc1_iterations = 2;
		zlut_minradius = 5.0f; zlut_maxradius = 60;
		zlut_angularsteps = 64;
		qi_iterations = 2;
		qi_radialsteps = qi_angularsteps = 64;
		qi_minradius = 5; qi_maxradius = 60;
	}
	int width, height;
	int numThreads, maxQueueSize;

	int xc1_profileLength;
	int xc1_profileWidth;
	int xc1_iterations;

	float zlut_minradius;
	float zlut_maxradius;
	int zlut_angularsteps;

	int qi_iterations;
	int qi_radialsteps, qi_angularsteps;
	float qi_minradius, qi_maxradius;
};
#pragma pack(pop)

class QueuedTracker
{
public:
	QueuedTracker() {}
	virtual ~QueuedTracker() {}

	virtual void Start () = 0;
	virtual void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane) = 0;
	virtual int PollFinished(LocalizationResult* results, int maxResults) = 0;

	virtual void SetZLUT(float* data,  int numLUTs, int planes, int res) = 0; // data can be zero to allocate ZLUT data
	virtual float* GetZLUT(int* planes, int *res, int *count) = 0; // delete[] memory afterwards

	// Debug stuff
	virtual float* GetDebugImage() { return 0; }

	virtual int GetJobCount() = 0;
	virtual int GetResultCount() = 0;

	virtual void GenerateTestImage(float* dstImg, float xp, float yp, float size, float photoncount) = 0;

	QTrkSettings cfg;
};

QueuedTracker* CreateQueuedTracker(QTrkSettings* s);

