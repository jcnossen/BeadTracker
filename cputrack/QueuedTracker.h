
// Defines the interface for trackers (both CPU and CUDA)

#pragma once

#include "std_incl.h" 
#include "threads.h"

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
	LocalizationJob(LocalizeType lt, uint frame, uint timestamp, uint zlutPlane, uint zlutIndex) :
		locType ( (uint)lt ), frame (frame), timestamp(timestamp), zlutPlane(zlutPlane), zlutIndex(zlutIndex) 
	{}
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
		xc1_profileLength = 128;
		xc1_profileWidth = 32;
		xc1_iterations = 2;
		zlut_minradius = 5.0f;
		zlut_angular_coverage = 0.7f;
		zlut_radial_coverage = 2.0f;
		zlut_roi_coverage = 1.0f;
		qi_iterations = 2;
		qi_minradius = 5;
		qi_angular_coverage = 0.7f;
		qi_radial_coverage = 2.0f; 
		qi_roi_coverage = 1.0f;
		qi_angstep_factor = 1.0f;
		cuda_device = -1;
		com_bgcorrection = 0.0f;
		gauss2D_iterations = 3;
	}
	int width, height;
	int numThreads;

#define QTrkCUDA_UseList -3   // Use list defined by SetCUDADevices
#define QTrkCUDA_UseAll -2
#define QTrkCUDA_UseBest -1
	// cuda_device < 0: use flags above
	// cuda_device >= 0: use as hardware device index
	int cuda_device;

	float com_bgcorrection; // 0.0f to disable
	int gauss2D_iterations;

	float zlut_minradius;
	float zlut_radial_coverage;
	float zlut_angular_coverage;
	float zlut_roi_coverage; // maxradius = ROI/2*roi_coverage

	int qi_iterations;
	float qi_minradius;
	float qi_radial_coverage;
	float qi_angular_coverage;
	float qi_roi_coverage;
	float qi_angstep_factor;

	int xc1_profileLength;
	int xc1_profileWidth;
	int xc1_iterations;
};
struct ROIPosition
{
	int x,y; // top-left coordinates. ROI is [ x .. x+w ; y .. y+h ]
};
#pragma pack(pop)

// Parameters computed from QTrkSettings
struct QTrkComputedConfig : public QTrkSettings
{
	QTrkComputedConfig() {}
	QTrkComputedConfig(const QTrkSettings& base) { *((QTrkSettings*)this)=base; Update(); }
	void Update();

	// Computed from QTrkSettings
	int zlut_radialsteps;
	int zlut_angularsteps;
	float zlut_maxradius;
	
	int qi_radialsteps;
	int qi_angstepspq;
	float qi_maxradius;
};

class AsyncScheduler;

// Abstract tracker interface, implementated by QueuedCUDATracker and QueuedCPUTracker
class QueuedTracker
{
public:
	QueuedTracker();
	virtual ~QueuedTracker();

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

	virtual int GetQueueLength(int *maxQueueLen=0) = 0;
	virtual bool IsIdle() = 0;

	virtual std::string GetProfileReport() { return ""; }

	QTrkComputedConfig cfg;

	void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint frame, uint timestamp, vector3f* initial, uint zlutIndex, uint zlutPlane);
	void ScheduleFrameAsync(uchar *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo);
	bool IsAsyncScheduleDone(uchar* imgptr);
	bool IsAsyncSchedulerIdle();
protected:
	AsyncScheduler* asyncScheduler;	
};

void CopyImageToFloat(uchar* data, int width, int height, int pitch, QTRK_PixelDataType pdt, float* dst);
QueuedTracker* CreateQueuedTracker(const QTrkComputedConfig& cc);
// if the tracker code is in a different DLL, you cannot call delete on the tracker instance. (DLLs do not share the memory heap with the host app)
void DestroyQueuedTracker(QueuedTracker* qtrk); 
void SetCUDADevices(int *devices, int numdev); // empty for CPU tracker

