/*
Labview API for the functionality in QueuedTracker.h
*/

#include "utils.h"
#include "labview.h"
#include "QueuedTracker.h"


CDLL_EXPORT void DLL_CALLCONV qtrk_compute_radial_profile(QueuedTracker* tracker, LVArray<float>** result, int angularSteps, float range, float* center)
{
	LVArray<float>* dst = *result;
	tracker->ComputeRadialProfile(&dst->elt[0], dst->dimSize, angularSteps, range, *(vector2f*)center);
}

CDLL_EXPORT void DLL_CALLCONV qtrk_set_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pZlut)
{
	LVArray3D<float>* zlut = *pZlut;

	int numLUTs = zlut->dimSizes[0];
	int planes = zlut->dimSizes[1];
	int res = zlut->dimSizes[2];
	
	tracker->SetZLUT(zlut->data, planes, res, numLUTs);
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
	qtrk->ScheduleLocalization(data, pitch, (QTRK_PixelDataType)pdt, (Localize2DType) locType, computeZ, id, zlutIndex);
}

CDLL_EXPORT int qtrk_jobcount(QueuedTracker* qtrk)
{
	return qtrk->GetJobCount();
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


