/*
Labview API for the functionality in QueuedTracker.h
*/

#include "utils.h"
#include "labview.h"
#include "QueuedTracker.h"

CDLL_EXPORT void DLL_CALLCONV qtrk_set_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pZlut)
{
	LVArray3D<float>* zlut = *pZlut;

	int numLUTs = zlut->dimSizes[0];
	int planes = zlut->dimSizes[1];
	int res = zlut->dimSizes[2];
	
	tracker->SetZLUT(zlut->elem, planes, res, numLUTs);
}

CDLL_EXPORT void DLL_CALLCONV qtrk_get_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pzlut)
{
	int dims[3];

	float* zlut = tracker->GetZLUT(&dims[0], &dims[1], &dims[2]);
	ResizeLVArray3D(pzlut, dims[0], dims[1], dims[2]);
	std::copy(zlut, zlut+(*pzlut)->numElem(), (*pzlut)->elem);
	delete[] zlut;
}

CDLL_EXPORT QueuedTracker* qtrk_create(QTrkSettings* settings, int startNow)
{
	QueuedTracker* tracker = CreateQueuedTracker(settings);
	if (startNow)
		tracker->Start();
	return tracker;
}

CDLL_EXPORT void qtrk_start(QueuedTracker* qtrk)
{
	qtrk->Start();
}


CDLL_EXPORT void qtrk_destroy(QueuedTracker* qtrk)
{
	delete qtrk;
}


CDLL_EXPORT void qtrk_queue_u16(QueuedTracker* qtrk, LVArray2D<ushort>** data, uint locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane)
{
	qtrk->ScheduleLocalization( (uchar*)(*data)->elem, sizeof(ushort)*(*data)->dimSizes[1], QTrkU16, (LocalizeType)locType, id, initialPos, zlutIndex, zlutPlane);
}

CDLL_EXPORT void qtrk_queue_u8(QueuedTracker* qtrk, LVArray2D<uchar>** data, uint locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane)
{
	qtrk->ScheduleLocalization( (*data)->elem, sizeof(uchar)*(*data)->dimSizes[1], QTrkU8, (LocalizeType) locType, id, initialPos, zlutIndex, zlutPlane);
}

CDLL_EXPORT void qtrk_queue_float(QueuedTracker* qtrk, LVArray2D<float>** data, uint locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane)
{
	qtrk->ScheduleLocalization( (uchar*) (*data)->elem, sizeof(float)*(*data)->dimSizes[1], QTrkFloat, (LocalizeType) locType, id, initialPos, zlutIndex, zlutPlane);
}

CDLL_EXPORT void test_array_passing(LVArray2D<float>** data, float* data2, int* len)
{
	int total=len[0]*len[1];
	for(int i=0;i<total;i++)
		dbgprintf("[%d] Data=%f, Data2=%f\n", i,(*data)->elem[i], data2[i]);
}

CDLL_EXPORT void qtrk_queue(QueuedTracker* qtrk, uchar* data, int pitch, uint pdt, uint locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane)
{
	qtrk->ScheduleLocalization(data, pitch, (QTRK_PixelDataType)pdt, (LocalizeType) locType, id, initialPos, zlutIndex, zlutPlane);
}

CDLL_EXPORT void qtrk_queue_array(QueuedTracker* qtrk, LVArray2D<uchar>** data, uint pdt, uint locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane)
{
	uint pitch;

	if (pdt == QTrkFloat) 
		pitch = sizeof(float);
	else if(pdt == QTrkU16) 
		pitch = 2;
	else pitch = 1;

	pitch *= (*data)->dimSizes[1]; // LVArray2D<uchar> type works for ushort and float as well
	dbgprintf("zlutindex: %d, zlutplane: %d\n", zlutIndex,zlutPlane);
	qtrk_queue(qtrk, (*data)->elem, pitch, pdt, locType, id, initialPos, zlutIndex, zlutPlane);
}



CDLL_EXPORT int qtrk_jobcount(QueuedTracker* qtrk)
{
	return qtrk->GetJobCount();
}


CDLL_EXPORT int qtrk_resultcount(QueuedTracker* qtrk)
{
	return qtrk->GetResultCount();
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

CDLL_EXPORT void DLL_CALLCONV qtrk_generate_test_image(QueuedTracker* tracker, LVArray2D<ushort>** image, float xp, float yp, float size, float photoncount)
{
	int w=tracker->cfg.width, h =tracker->cfg.height;
	ResizeLVArray2D(image, h,w);
	
	float *d = new float[w*h];
	tracker->GenerateTestImage(d, xp, yp, size, photoncount );
	floatToNormalizedUShort((*image)->elem, d, w,h);
	delete[] d;
}

CDLL_EXPORT void DLL_CALLCONV qtrk_generate_image_from_lut(LVArray2D<float>** image, LVArray2D<float>** lut, float LUTradius, vector2f* position, float z, float M, float photonCountPP)
{
	ImageData img((*image)->elem, (*image)->dimSizes[1], (*image)->dimSizes[0]);
	ImageData zlut((*lut)->elem, (*lut)->dimSizes[1], (*lut)->dimSizes[0]);

	GenerateImageFromLUT(&img, &zlut, LUTradius, *position, z, M);
	img.normalize();
	if(photonCountPP>0)
		ApplyPoissonNoise(img, photonCountPP);
}


CDLL_EXPORT void DLL_CALLCONV qtrk_read_jpeg_from_file(const char* filename, LVArray2D<float>** dstImage)
{
	int w,h;
	uchar* data;
	
	std::vector<uchar> buf = ReadToByteBuffer(filename);
	ReadJPEGFile(&buf[0], buf.size(), &data,&w,&h);

	if ( (*dstImage)->dimSizes[0] != h || (*dstImage)->dimSizes[1] != w )
		ResizeLVArray2D(dstImage, h, w);

	memcpy( (*dstImage)->elem, data, w*h );
	delete[] data;
}



