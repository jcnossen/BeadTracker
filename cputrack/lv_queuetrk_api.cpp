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


CDLL_EXPORT QueuedTracker* qtrk_create(QTrkSettings* settings)
{
	return CreateQueuedTracker(settings);
}

CDLL_EXPORT void qtrk_destroy(QueuedTracker* qtrk)
{
	delete qtrk;
}

CDLL_EXPORT void qtrk_queue(QueuedTracker* qtrk, uchar* data, int pitch, uint pdt, uint locType, bool computeZ, uint id, uint zlutIndex)
{
	if (computeZ) {
		locType |= LocalizeZ;
	}

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

CDLL_EXPORT int qtrk_get_results(QueuedTracker* qtrk, LocalizationResult* results, int maxResults)
{
	return qtrk->PollFinished(results, maxResults);
}


CDLL_EXPORT void DLL_CALLCONV generate_test_image(Image *img, int w, int h, float xp, float yp, float size, float photoncount)
{
	try {
		float *d = new float[w*h];
		GenerateTestImage(d, w, h, xp, yp, size, photoncount);
		ushort* intd = floatToNormalizedUShort(d, w,h);

		imaqArrayToImage(img, intd, w,h);
		delete[] d;
		delete[] intd;
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
	}
}


