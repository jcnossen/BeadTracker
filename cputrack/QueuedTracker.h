
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
struct QISettings {
	QISettings() {
		iterations = 2;
		radialsteps = angularsteps = 64;
		minradius = 5; maxradius = 60;
	}
	int iterations;
	int radialsteps, angularsteps;
	float minradius, maxradius;
};
struct XCor1DSettings {
	XCor1DSettings() {
		profileLength = 128; 
		profileWidth = 32;
		iterations = 2;
	}
	int profileLength;
	int profileWidth;
	int iterations;
};
struct ZLUTSettings {
	ZLUTSettings() {
		minradius = 5.0f; maxradius = 60;
		angularsteps = 64;
	}
	float minradius;
	float maxradius;
	int angularsteps;
};

struct QTrkMainSettings {
	QTrkMainSettings() {
		width = height = 150;
		numThreads = -1;
		maxQueueSize = 200;
	}
	int width, height;
	int numThreads, maxQueueSize;
};

// DONT CHANGE, Mapped to labview clusters (QTrkSettings.ctl)!
struct QTrkSettings : public QTrkMainSettings {
	QISettings qi;
	XCor1DSettings xc1;
	ZLUTSettings zlut;
};
#pragma pack(pop)

class QueuedTracker
{
public:
	QueuedTracker() {}
	virtual ~QueuedTracker() {}

	virtual void Start () = 0;
	virtual void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, uint zlutIndex=0) = 0;
	virtual int PollFinished(LocalizationResult* results, int maxResults) = 0;

	virtual void SetZLUT(float* data, int planes, int res, int numLUTs) = 0;
	virtual void ComputeRadialProfile(float *image, int width, int height, float* dst, int profileLength, vector2f center) = 0;

	// Debug stuff
	virtual float* GetDebugImage() { return 0; }

	virtual int GetJobCount() = 0;
	virtual int GetResultCount() = 0;

	virtual void GenerateTestImage(float* dstImg, float xp, float yp, float size, float photoncount) = 0;

	QTrkSettings cfg;
};

QueuedTracker* CreateQueuedTracker(QTrkSettings* s);

