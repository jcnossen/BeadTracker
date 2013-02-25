/*
Labview API for the functionality in QueuedTracker.h
*/
#include "std_incl.h"
#include "utils.h"
#include "labview.h"
#include "QueuedTracker.h"
#include "threads.h" 


#include "lv_prolog.h"
struct QueueImageParams {
	uint locType;
	uint frame;
	vector3f initialPos;
	uint zlutIndex; // or bead#
	uint zlutPlane; // for ZLUT building

	LocalizeType LocType()  { return (LocalizeType)locType; }
};
#include "lv_epilog.h"


static Threads::Mutex trackerListMutex;
static std::vector<QueuedTracker*> trackerList;

CDLL_EXPORT void DLL_CALLCONV qtrk_free_all()
{
	trackerListMutex.lock();
	DeleteAllElems(trackerList);
	trackerListMutex.unlock();
}


MgErr FillErrorCluster(MgErr err, const char *message, ErrorCluster *error)
{
	if (err)
	{
		int msglen = strlen(message);
		error->status = LVBooleanTrue;
		error->code = err;
		err = NumericArrayResize(uB, 1, (UHandle*)&(error->message), msglen);
		if (!err)
		{
			MoveBlock(message, LStrBuf(*error->message), msglen);
			LStrLen(*error->message) = msglen;
		} 
	}
	return err;
}

void ArgumentErrorMsg(ErrorCluster* e, const std::string& msg) {
	FillErrorCluster(mgArgErr, msg.c_str(), e);
}

CDLL_EXPORT void DLL_CALLCONV qtrk_set_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pZlut, LVArray<float>** zcmpWindow, ErrorCluster* e)
{
	LVArray3D<float>* zlut = *pZlut;

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
	
	tracker->SetZLUT(zlut->elem, numLUTs, planes, res, zcmp);
}

CDLL_EXPORT void DLL_CALLCONV qtrk_get_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pzlut)
{
	int dims[3];

	float* zlut = tracker->GetZLUT(&dims[0], &dims[1], &dims[2]);
	ResizeLVArray3D(pzlut, dims[0], dims[1], dims[2]);
	memcpy((*pzlut)->elem, zlut, sizeof(float)*(*pzlut)->numElem());
	delete[] zlut;
}

CDLL_EXPORT QueuedTracker* qtrk_create(QTrkSettings* settings, ErrorCluster* e)
{
	QueuedTracker* tracker = 0;
	try {
		tracker = CreateQueuedTracker(settings);

		trackerListMutex.lock();
		trackerList.push_back(tracker);
		trackerListMutex.unlock();

		tracker->Start();
	} catch(const std::runtime_error &exc) {
		FillErrorCluster(kAppErrorBase, exc.what(), e );
	}
	return tracker;
}


CDLL_EXPORT void qtrk_destroy(QueuedTracker* qtrk)
{
	trackerListMutex.lock();
	trackerList.erase(std::find(trackerList.begin(),trackerList.end(), qtrk));
	trackerListMutex.unlock();

	delete qtrk;
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

CDLL_EXPORT void qtrk_queue_u16(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<ushort>** data, QueueImageParams* params)
{
	if (CheckImageInput(qtrk, data, error))
		qtrk->ScheduleLocalization( (uchar*)(*data)->elem, sizeof(ushort)*(*data)->dimSizes[1], QTrkU16, params->LocType(), params->frame, &params->initialPos, params->zlutIndex, params->zlutPlane);
}

CDLL_EXPORT void qtrk_queue_u8(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<uchar>** data, QueueImageParams* params)
{
	if (CheckImageInput(qtrk, data, error))
		qtrk->ScheduleLocalization( (*data)->elem, sizeof(uchar)*(*data)->dimSizes[1], QTrkU8, params->LocType(), params->frame, &params->initialPos, params->zlutIndex, params->zlutPlane);
}

CDLL_EXPORT void qtrk_queue_float(QueuedTracker* qtrk, ErrorCluster* error, LVArray2D<float>** data, QueueImageParams* params)
{
	if (CheckImageInput(qtrk, data, error))
		qtrk->ScheduleLocalization( (uchar*) (*data)->elem, sizeof(float)*(*data)->dimSizes[1], QTrkFloat, params->LocType(), params->frame, &params->initialPos, params->zlutIndex, params->zlutPlane);
}


CDLL_EXPORT void qtrk_queue_pitchedmem(QueuedTracker* qtrk, uchar* data, int pitch, uint pdt, QueueImageParams* params)
{
	qtrk->ScheduleLocalization(data, pitch, (QTRK_PixelDataType)pdt, params->LocType(), params->frame, &params->initialPos, params->zlutIndex, params->zlutPlane);
}

CDLL_EXPORT void qtrk_queue_array(QueuedTracker* qtrk,  ErrorCluster* error,LVArray2D<uchar>** data,uint pdt,  QueueImageParams* params)
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
	qtrk_queue_pitchedmem(qtrk, (*data)->elem, pitch, pdt, params);
}


CDLL_EXPORT void qtrk_queue_frame(QueuedTracker* qtrk, uchar* image, int pitch, int w,int h, 
	uint pdt, ROIPosition* pos, int numROI, uint locType, uint frame, uint zlutPlane, bool async)
{
	qtrk->ScheduleFrame(image, pitch, w,h, pos, numROI, (QTRK_PixelDataType)pdt, (LocalizeType)locType, frame, zlutPlane, async);
}

CDLL_EXPORT void qtrk_wait_for_queue_frame(QueuedTracker* qtrk, uchar* image)
{
	qtrk->WaitForScheduleFrame(image);
}


CDLL_EXPORT void qtrk_clear_results(QueuedTracker* qtrk)
{
	qtrk->ClearResults();
}


CDLL_EXPORT int qtrk_hasfullqueue(QueuedTracker* qtrk) 
{
	if (!qtrk)
		return 0;

	return qtrk->IsQueueFilled() ? 1 : 0;
}

CDLL_EXPORT int qtrk_resultcount(QueuedTracker* qtrk)
{
	if (!qtrk)
		return 0;
	return qtrk->GetResultCount();
}

CDLL_EXPORT void qtrk_flush(QueuedTracker* qtrk)
{
	if (qtrk)
		qtrk->Flush();
}

static bool compareResultsByID(const LocalizationResult& a, const LocalizationResult& b) {
	return a.id<b.id;
}

CDLL_EXPORT int qtrk_get_results(QueuedTracker* qtrk, LocalizationResult* results, int maxResults, int sortByID)
{
	int resultCount = qtrk->PollFinished(results, maxResults);

	if (sortByID) {
		std::sort(results, results+resultCount, compareResultsByID);
	}

	return resultCount;
}

CDLL_EXPORT int qtrk_idle(QueuedTracker* qtrk)
{
	return qtrk->IsIdle() ? 1 : 0;
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



CDLL_EXPORT void test_struct_pass(ErrorCluster* e, QueueImageParams* d1)
{

}


