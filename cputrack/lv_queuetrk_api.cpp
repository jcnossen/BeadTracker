/*
Labview API for the functionality in QueuedTracker.h
*/

#include "utils.h"
#include "labview.h"
#include "QueuedTracker.h"


CDLL_EXPORT void DLL_CALLCONV qtrk_compute_radial_profile(QueuedTracker* tracker, LVArray2D<float>** p_image, LVArray<float>** result, vector2f* center)
{
	LVArray<float>* dst = *result;
	LVArray2D<float>* image = *p_image;

	tracker->ComputeRadialProfile(image->elem, image->dimSizes[1], image->dimSizes[0], dst->elem, dst->dimSize, *center);
}

CDLL_EXPORT void DLL_CALLCONV qtrk_set_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pZlut)
{
	LVArray3D<float>* zlut = *pZlut;

	int numLUTs = zlut->dimSizes[0];
	int planes = zlut->dimSizes[1];
	int res = zlut->dimSizes[2];
	
	tracker->SetZLUT(zlut->elem, planes, res, numLUTs);
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

CDLL_EXPORT void qtrk_queue(QueuedTracker* qtrk, uchar* data, int pitch, uint pdt, uint locType, uint computeZ, uint id, uint zlutIndex)
{
	if (computeZ) {
		locType |= LocalizeZ;
	} else
		zlutIndex = 0;

	qtrk->ScheduleLocalization(data, pitch, (QTRK_PixelDataType)pdt, (LocalizeType) locType, id, zlutIndex);
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

CDLL_EXPORT void DLL_CALLCONV generate_image_from_lut(LVArray2D<float>** image, LVArray2D<float>** lut, float LUTradius, vector2f* position, float z, float M, float photonCountPP)
{
	ImageData img((*image)->elem, (*image)->dimSizes[1], (*image)->dimSizes[0]);
	ImageData zlut((*lut)->elem, (*lut)->dimSizes[1], (*lut)->dimSizes[0]);

	GenerateImageFromLUT(&img, &zlut, LUTradius, *position, z, M);
	img.normalize();
	if(photonCountPP>0)
		ApplyPoissonNoise(img, photonCountPP);
}

