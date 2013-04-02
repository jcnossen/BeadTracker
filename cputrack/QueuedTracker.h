
// Defines the interface for trackers (both CPU and CUDA)

#pragma once

#include "std_incl.h" 

enum LocalizeType {
	// Flags for selecting 2D localization type
	LocalizeOnlyCOM = 0, // use only COM
	LocalizeXCor1D = 1, // COM+XCor1D
	LocalizeQI = 2, // COM+QI
	LocalizeXCor2D = 3,   // XCor2D
	LocalizeGaussian2D = 4, // 2D Gaussian localization

	Localize2DMask = 15,
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
struct LocalizationJob {
	LocalizationJob() {
		locType=frame=timestamp=zlutIndex=zlutPlane=0; 
	}
	uint locType;
	uint frame, timestamp;
	uint zlutIndex; // or bead#
	uint zlutPlane; // for ZLUT building
	vector3f initialPos;
	LocalizeType LocType() const { return (LocalizeType)locType; }
};

// DONT CHANGE, Mapped to labview clusters!
struct LocalizationResult {
	LocalizationJob job;
	vector3f pos;
	vector2f pos2D() { return vector2f(pos.x,pos.y); }
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
		zlut_angularsteps = 128;
		zlut_radialsteps = 32;
		qi_iterations = 2;
		qi_radialsteps = 32; 
		qi_angsteps_per_quadrant = 32;
		qi_minradius = 5; qi_maxradius = 60;
		cuda_device = -1;
		com_bgcorrection = 0.0f;
		gauss2D_iterations = 3;
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
	int qi_radialsteps; 
	int qi_angsteps_per_quadrant;// Per quadrant
	float qi_minradius, qi_maxradius;

#define QTrkCUDA_UseList -3   // Use list defined by SetCUDADevices
#define QTrkCUDA_UseAll -2
#define QTrkCUDA_UseBest -1
	// cuda_device < 0: use flags above
	// cuda_device >= 0: use as hardware device index
	int cuda_device;

	float com_bgcorrection; // 0.0f to disable
	int zlut_radialsteps;
	int gauss2D_iterations;
};
struct ROIPosition
{
	int x,y; // top-left coordinates. ROI is [ x .. x+w ; y .. y+h ]
};
#pragma pack(pop)

// Abstract tracker interface, implementated by QueuedCUDATracker and QueuedCPUTracker
class QueuedTracker
{
public:
	QueuedTracker() {}
	virtual ~QueuedTracker() {}

	// Frame and timestamp are ignored by tracking code itself, but usable for the calling code
	// Pitch: Distance in bytes between two successive rows of pixels (e.g. address of (0,0) -  address of (0,1) )
	// ZlutIndex: Which ZLUT to use for ComputeZ/BuildZLUT
	virtual void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo) = 0;
	virtual void ClearResults() = 0;
	virtual void Flush() = 0; // stop waiting for more jobs to do, and just process the current batch

	// Schedule an entire frame at once, allowing for further optimizations
	virtual void ScheduleFrame(uchar *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo) = 0;
	
	// data can be zero to allocate ZLUT data. zcmp has to have 'res' elements
	virtual void SetZLUT(float* data, int count, int planes, float* zcmp=0) = 0; 
	virtual float* GetZLUT(int *count=0, int* planes=0) = 0; // delete[] memory afterwards
	virtual int GetResultCount() = 0;
	virtual int PollFinished(LocalizationResult* results, int maxResults) = 0;

	virtual bool IsQueueFilled() = 0;
	virtual bool IsIdle() = 0;

	virtual std::string GetProfileReport() { return ""; }

	QTrkSettings cfg;

	void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint frame, uint timestamp, vector3f* initial, uint zlutIndex, uint zlutPlane) {
		LocalizationJob j;
		j.frame= frame;
		j.locType = locType;
		j.timestamp = timestamp;
		if (initial) j.initialPos = *initial;
		j.zlutIndex = zlutIndex;
		j.zlutPlane = zlutPlane;
		ScheduleLocalization(data,pitch,pdt,&j);
	}

};


void CopyImageToFloat(uchar* data, int width, int height, int pitch, QTRK_PixelDataType pdt, float* dst);
QueuedTracker* CreateQueuedTracker(QTrkSettings* s);
void SetCUDADevices(std::vector<int> devices); // empty for CPU tracker

