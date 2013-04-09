/*
Labview API for the functionality in QueuedTracker.h
*/
#include "std_incl.h"
#include "utils.h"
#include "labview.h"
#include "QueuedTracker.h"
#include "threads.h" 


static Threads::Mutex trackerListMutex;
static std::vector<QueuedTracker*> trackerList;

CDLL_EXPORT void DLL_CALLCONV qtrk_free_all()
{
	trackerListMutex.lock();
	DeleteAllElems(trackerList);
	trackerListMutex.unlock();
}


void SetLVString (LStrHandle str, const char *text)
{
	int msglen = strlen(text);
	MgErr err = NumericArrayResize(uB, 1, (UHandle*)&str, msglen);
	if (!err)
	{
		MoveBlock(text, LStrBuf(*str), msglen);
		LStrLen(*str) = msglen;
	}
}

MgErr FillErrorCluster(MgErr err, const char *message, ErrorCluster *error)
{
	if (err)
	{
		error->status = LVBooleanTrue;
		error->code = err;
		SetLVString ( error->message, message );
	}
	return err;
}

void ArgumentErrorMsg(ErrorCluster* e, const std::string& msg) {
	FillErrorCluster(mgArgErr, msg.c_str(), e);
}

bool ValidateTracker(QueuedTracker* tracker, ErrorCluster* e, const char *funcname)
{
	if (std::find(trackerList.begin(), trackerList.end(), tracker)==trackerList.end()) {
		ArgumentErrorMsg(e, SPrintf("QTrk C++ function %s called with invalid tracker pointer: %p", funcname, tracker));
		return false;
	}
	return true;
}

CDLL_EXPORT void DLL_CALLCONV qtrk_set_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pZlut, LVArray<float>** zcmpWindow, ErrorCluster* e)
{
	LVArray3D<float>* zlut = *pZlut;

	if (ValidateTracker(tracker,e, "set ZLUT")) {

		int numLUTs = zlut->dimSizes[0];
		int planes = zlut->dimSizes[1];
		int res = zlut->dimSizes[2];

		dbgprintf("Setting ZLUT size: %d beads, %d planes, %d radialsteps\n", numLUTs, planes, res);

		float* zcmp = 0;
		if (zcmpWindow && (*zcmpWindow)->dimSize > 0) {
			if ( (*zcmpWindow)->dimSize != res)
				ArgumentErrorMsg(e, SPrintf("Z Compare window should have the same resolution as the ZLUT (%d elements)", res));
			else
				zcmp = (*zcmpWindow)->elem;
		}

		if (numLUTs * planes * res == 0) {
			tracker->SetZLUT(0, 0, 0);
		} else {
			if (res != tracker->cfg.zlut_radialsteps)
				ArgumentErrorMsg(e, SPrintf("set_ZLUT: 3rd dimension should have size of zlut_radialsteps (%d instead of %d)", tracker->cfg.zlut_radialsteps, res));
			else 
				tracker->SetZLUT(zlut->elem, numLUTs, planes, zcmp);
		}
	}
}

CDLL_EXPORT void DLL_CALLCONV qtrk_get_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pzlut, ErrorCluster* e)
{
	if (ValidateTracker(tracker, e, "get ZLUT")) {
		int dims[3];
		dims[2] = tracker->cfg.zlut_radialsteps;
		float* zlut = tracker->GetZLUT(&dims[0], &dims[1]);
		if (zlut) {
			ResizeLVArray3D(pzlut, dims[0], dims[1], dims[2]);
			memcpy((*pzlut)->elem, zlut, sizeof(float)*(*pzlut)->numElem());
			delete[] zlut;
		}
	}
}

CDLL_EXPORT QueuedTracker* qtrk_create(QTrkSettings* settings, ErrorCluster* e)
{
	QueuedTracker* tracker = 0;
	try {
		tracker = CreateQueuedTracker(settings);

		trackerListMutex.lock();
		trackerList.push_back(tracker);
		trackerListMutex.unlock();
	} catch(const std::runtime_error &exc) {
		FillErrorCluster(kAppErrorBase, exc.what(), e );
	}
	return tracker;
}


CDLL_EXPORT void qtrk_destroy(QueuedTracker* qtrk, ErrorCluster* error)
{
	trackerListMutex.lock();

	auto pos = std::find(trackerList.begin(),trackerList.end(),qtrk);
	if (pos == trackerList.end()) {
		ArgumentErrorMsg(error, SPrintf( "Trying to call qtrk_destroy with invalid qtrk pointer %p", qtrk));
		qtrk = 0;
	}
	else
		trackerList.erase(pos);

	trackerListMutex.unlock();

	if(qtrk) delete qtrk;
}

template<typename T>
bool CheckImageInput(QueuedTracker* qtrk, LVArray2D<T> **data, ErrorCluster  *error)
{
	if (!data) {
		ArgumentErrorMsg(error, "Image data array is empty");
		return false;
	} else if( (*data)->dimSizes[1] != qtrk->cfg.width || (*data)->dimSizes[0] != qtrk->cfg.height ) {
		ArgumentErrorMsg(error, SPrintf( "Image data array has wrong size (%d,%d). Should be: (%d,%d)", (*data)->dimSizes[1], (*data)->dimSizes[0], qtrk->cfg.width, qtrk->cfg.height));
		return false;
	}
	return true;
}

CDLL_EXPORT void qtrk_queue_u16(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<ushort>** data, const LocalizationJob *jobInfo)
{
	if (CheckImageInput(qtrk, data, error)) 
	{
		qtrk->ScheduleLocalization( (uchar*)(*data)->elem, sizeof(ushort)*(*data)->dimSizes[1], QTrkU16, jobInfo);
	}
}

CDLL_EXPORT void qtrk_queue_u8(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<uchar>** data, const LocalizationJob *jobInfo)
{
	if (CheckImageInput(qtrk, data, error))
	{
		qtrk->ScheduleLocalization( (*data)->elem, sizeof(uchar)*(*data)->dimSizes[1], QTrkU8, jobInfo);
	}
}

CDLL_EXPORT void qtrk_queue_float(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<float>** data, const LocalizationJob *jobInfo)
{
	if (CheckImageInput(qtrk, data, error))
	{
		qtrk->ScheduleLocalization( (uchar*) (*data)->elem, sizeof(float)*(*data)->dimSizes[1], QTrkFloat, jobInfo);
	}
}


CDLL_EXPORT void qtrk_queue_pitchedmem(QueuedTracker* qtrk, uchar* data, int pitch, uint pdt, const LocalizationJob *jobInfo)
{
	qtrk->ScheduleLocalization(data, pitch, (QTRK_PixelDataType)pdt, jobInfo);
}

CDLL_EXPORT void qtrk_queue_array(QueuedTracker* qtrk,  ErrorCluster* error,LVArray2D<uchar>** data,uint pdt, const LocalizationJob *jobInfo)
{
	uint pitch;

	if (pdt == QTrkFloat) 
		pitch = sizeof(float);
	else if(pdt == QTrkU16) 
		pitch = 2;
	else pitch = 1;

	if (!CheckImageInput(qtrk, data, error))
		return;

	pitch *= (*data)->dimSizes[1]; // LVArray2D<uchar> type works for ushort and float as well
//	dbgprintf("zlutindex: %d, zlutplane: %d\n", zlutIndex,zlutPlane);
	qtrk_queue_pitchedmem(qtrk, (*data)->elem, pitch, pdt, jobInfo);
}


CDLL_EXPORT void qtrk_queue_frame(QueuedTracker* qtrk, uchar* image, int pitch, int w,int h, 
	uint pdt, ROIPosition* pos, int numROI, const LocalizationJob *jobInfo)
{
	qtrk->ScheduleFrame(image, pitch, w,h, pos, numROI, (QTRK_PixelDataType)pdt, jobInfo);
}


CDLL_EXPORT void qtrk_clear_results(QueuedTracker* qtrk, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "clear results")) {
		qtrk->ClearResults();
	}
}


CDLL_EXPORT int qtrk_hasfullqueue(QueuedTracker* qtrk, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "fullqueue"))
		return qtrk->IsQueueFilled() ? 1 : 0;
	return 0;
}

CDLL_EXPORT int qtrk_resultcount(QueuedTracker* qtrk, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "resultcount")) {
		return qtrk->GetResultCount();
	} 
	return 0;
}

CDLL_EXPORT void qtrk_flush(QueuedTracker* qtrk, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "flush")) {
		qtrk->Flush();
	}
}

CDLL_EXPORT int qtrk_get_results(QueuedTracker* qtrk, LocalizationResult* results, int maxResults, int sortByID, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "get_results")) {
		int resultCount = qtrk->PollFinished(results, maxResults);

		if (sortByID) {
			std::sort(results, results+resultCount, [](decltype(*results) a, decltype(*results) b) { return a.job.frame<b.job.frame; } );
		}

		return resultCount;
	} 
	return 0;
}


CDLL_EXPORT int qtrk_idle(QueuedTracker* qtrk, ErrorCluster* e)
{
	if (ValidateTracker(qtrk, e, "is_idle"))
		return qtrk->IsIdle() ? 1 : 0;
	return 0;
}

CDLL_EXPORT void DLL_CALLCONV qtrk_generate_image_from_lut(LVArray2D<float>** image, LVArray2D<float>** lut, 
					float *LUTradii, vector2f* position, float z, float M, float sigma_noise)
{
	ImageData img((*image)->elem, (*image)->dimSizes[1], (*image)->dimSizes[0]);
	ImageData zlut((*lut)->elem, (*lut)->dimSizes[1], (*lut)->dimSizes[0]);

	GenerateImageFromLUT(&img, &zlut, LUTradii[0], LUTradii[1], *position, z, M);
	//img.normalize();
	if(sigma_noise>0)
		ApplyGaussianNoise(img, sigma_noise);
}


CDLL_EXPORT void qtrk_dump_memleaks()
{
#ifdef USE_MEMDBG
	_CrtDumpMemoryLeaks();
#endif
}

CDLL_EXPORT void qtrk_get_profile_report(QueuedTracker* qtrk, LStrHandle str)
{
	SetLVString(str, qtrk->GetProfileReport().c_str());
}


#ifdef CUDA_TRACK

#include "cuda_runtime.h"

CDLL_EXPORT void qtrkcuda_set_device_list(LVArray<int>** devices)
{
	std::vector<int> devlist( (*devices)->elem, (*devices)->elem + (*devices)->dimSize );
	SetCUDADevices(devlist);
}

static bool CheckCUDAErrorLV(cudaError err, ErrorCluster* e)
{
	if (err != cudaSuccess) {
		const char* errstr = cudaGetErrorString(err);
		FillErrorCluster(kAppErrorBase, SPrintf("CUDA error: %s", errstr).c_str(), e);
		return false;
	}
	return true;
}

#pragma pack(push,1)
struct CUDADeviceInfo 
{
	LStrHandle name;
	int clockRate;
	int multiProcCount;
	int major, minor;
};
#pragma pack(pop)

CDLL_EXPORT int qtrkcuda_device_count(ErrorCluster* e) {
	int c;
	if (CheckCUDAErrorLV(cudaGetDeviceCount(&c), e)) {
		return c;
	}
	return 0;
}

CDLL_EXPORT void qtrkcuda_get_device(int device, CUDADeviceInfo *info, ErrorCluster* e)
{
	cudaDeviceProp prop;
	if (CheckCUDAErrorLV(cudaGetDeviceProperties(&prop, device), e)) {
		info->multiProcCount = prop.multiProcessorCount;
		info->clockRate = prop.clockRate;
		info->major = prop.major;
		info->minor = prop.minor;
		SetLVString(info->name, prop.name);
	}
}

#else


CDLL_EXPORT int qtrkcuda_device_count(ErrorCluster* e) { return 0; }
CDLL_EXPORT void qtrkcuda_get_device(int device, void *info, ErrorCluster* e) {}

#endif

