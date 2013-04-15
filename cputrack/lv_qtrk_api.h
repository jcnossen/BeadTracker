
#include "std_incl.h"
#include "labview.h"

#include "QueuedTracker.h"

CDLL_EXPORT void DLL_CALLCONV qtrk_set_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pZlut, LVArray<float>** zcmpWindow, ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV qtrk_get_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pzlut, ErrorCluster* e);
CDLL_EXPORT QueuedTracker* DLL_CALLCONV qtrk_create(QTrkSettings* settings, ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV qtrk_destroy(QueuedTracker* qtrk, ErrorCluster* error);
CDLL_EXPORT void DLL_CALLCONV qtrk_queue_u16(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<ushort>** data, const LocalizationJob *jobInfo);
CDLL_EXPORT void DLL_CALLCONV qtrk_queue_u8(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<uchar>** data, const LocalizationJob *jobInfo);
CDLL_EXPORT void DLL_CALLCONV qtrk_queue_float(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<float>** data, const LocalizationJob *jobInfo);
CDLL_EXPORT void DLL_CALLCONV qtrk_queue_pitchedmem(QueuedTracker* qtrk, uchar* data, int pitch, uint pdt, const LocalizationJob *jobInfo);
CDLL_EXPORT void DLL_CALLCONV qtrk_queue_array(QueuedTracker* qtrk,  ErrorCluster* error,LVArray2D<uchar>** data,uint pdt, const LocalizationJob *jobInfo);


enum QueueFrameFlags {
	QFF_ReadTimestampFromFrame = 1,
	QFF_ReadTimestampFromFrameRev = 2,
	QFF_Force32Bit = 0x7fffffff
};

CDLL_EXPORT uint DLL_CALLCONV qtrk_queue_frame(QueuedTracker* qtrk, uchar* image, int pitch, int w,int h, uint pdt, ROIPosition* pos, int numROI, const LocalizationJob *pJobInfo, QueueFrameFlags flags);
CDLL_EXPORT void DLL_CALLCONV qtrk_clear_results(QueuedTracker* qtrk, ErrorCluster* e);
CDLL_EXPORT int DLL_CALLCONV qtrk_hasfullqueue(QueuedTracker* qtrk, ErrorCluster* e);
CDLL_EXPORT int DLL_CALLCONV qtrk_resultcount(QueuedTracker* qtrk, ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV qtrk_flush(QueuedTracker* qtrk, ErrorCluster* e);
CDLL_EXPORT int DLL_CALLCONV qtrk_get_results(QueuedTracker* qtrk, LocalizationResult* results, int maxResults, int sortByID, ErrorCluster* e);
CDLL_EXPORT int DLL_CALLCONV qtrk_idle(QueuedTracker* qtrk, ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV qtrk_generate_image_from_lut(LVArray2D<float>** image, LVArray2D<float>** lut, float *LUTradii, vector2f* position, float z, float M, float sigma_noise);
CDLL_EXPORT void DLL_CALLCONV qtrk_dump_memleaks();
CDLL_EXPORT void qtrk_get_profile_report(QueuedTracker* qtrk, LStrHandle str);


					
#pragma pack(push,1)
struct CUDADeviceInfo 
{
	LStrHandle name;
	int clockRate;
	int multiProcCount;
	int major, minor;
};
#pragma pack(pop)

CDLL_EXPORT int DLL_CALLCONV qtrkcuda_device_count(ErrorCluster* e);
CDLL_EXPORT void DLL_CALLCONV qtrkcuda_set_device_list(LVArray<int>** devices);

