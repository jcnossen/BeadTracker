
#include <Windows.h>

#include "nivision.h"
#include "extcode.h"
#include "niimaq.h"
#include "random_distr.h"

/* lv_prolog.h and lv_epilog.h set up the correct alignment for LabVIEW data. */
#include "lv_prolog.h"
struct LVFloatArray {
	int32_t dimSize;
	float elt[1];
};
typedef LVFloatArray **ppFloatArray;
struct LVFloatArray2 {
	int32_t dimSizes[2];
	float data[1];
	float& elem(int col, int row) {
		return data[row*dimSizes[0]+col];
	}
};
typedef LVFloatArray2 **ppFloatArray2;
#include "lv_epilog.h"

#include "tracker.h"
#include "MThreadQueue.h"


void saveImage(float* data, uint w, uint h, const char* filename)
{
	ushort* d = floatToNormalizedUShort(data,w,h);
	Image* dst = imaqCreateImage(IMAQ_IMAGE_U16, 0);
	imaqSetImageSize(dst, w, h);
	imaqArrayToImage(dst, d, w, h);
	delete[] d;

	ImageInfo info;
	imaqGetImageInfo(dst, &info);
	int success = imaqWriteFile(dst, filename, 0);
	if (!success) {
		char *errStr = imaqGetErrorText(imaqGetLastError());
		std::string msg = SPrintf("IMAQ WriteFile error: %s\n", errStr);
		imaqDispose(errStr);
		dbgout(msg);
	}
	imaqDispose(dst);
}

CDLL_EXPORT void DLL_CALLCONV generate_test_image(Image *img, int w, int h, float xp, float yp, float size, float photoncount)
{
	try {
		float S = size;
		float *d = new float[w*h];
		for (int y=0;y<h;y++) {
			for (int x=0;x<w;x++) {
				float X = x - xp;
				float Y = y - yp;
				float r = sqrtf(X*X+Y*Y)+1;
				float v = sinf(r*S/5.0f) * expf(-r*r/S*0.01f);
				d[y*w+x] = v;
			}
		}

		ushort* result = floatToNormalizedUShort(d, w, h);
		if (photoncount != 0.0f) {
			for (int k=0;k<w*h;k++)
				result[k] = rand_poisson((float)result[k] * photoncount / (float) (1<<16));
		}

		imaqArrayToImage(img, result, w,h);
		delete[] result;
		delete[] d;
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
	}
}

CDLL_EXPORT Tracker* DLL_CALLCONV create_tracker(uint w, uint h, uint xcorw)
{
	try {
		Sleep(300);
		return CreateTrackerInstance(w,h,xcorw);
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
		return 0;
	}
}

CDLL_EXPORT void DLL_CALLCONV destroy_tracker(Tracker* tracker)
{
	try {
		delete tracker;
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
	}
}

template<typename T>
void copyToLVArray (ppFloatArray r, const std::vector<T>& a)
{
	size_t wantedSize = sizeof(float)*a.size();

	NumericArrayResize(9 /* codes for SGL type */, 1, (UHandle*)&r, wantedSize);

	LVFloatArray* dst = *r;
	dst->dimSize = a.size();
	size_t len = min((size_t) dst->dimSize, a.size () );
//	dbgout(SPrintf("copying %d elements to Labview array\n", len));
	for (size_t i=0;i<a.size();i++)
		dst->elt[i] = a[i];
}

CDLL_EXPORT void DLL_CALLCONV copy_crosscorrelation_result(Tracker* tracker, ppFloatArray x_result, ppFloatArray y_result, ppFloatArray x_xc, ppFloatArray y_xc)
{
	try {
		std::vector<xcor_t> xprof, yprof, xconv, yconv;
		if (tracker->GetLastXCorProfiles(xprof, yprof, xconv, yconv)) {
			if (x_result) copyToLVArray (x_result, xprof);
			if (y_result) copyToLVArray (y_result, yprof);
			if (x_xc) copyToLVArray (x_xc, xconv);
			if (y_xc) copyToLVArray (y_xc, yconv);
		}
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
	}
}

CDLL_EXPORT void DLL_CALLCONV localize_image(Tracker* tracker, Image* img, float* COM, float* xcor, float* storedMedian, Image* dbgImg, int xcor_iterations)
{
	try {
		ImageInfo info;
		imaqGetImageInfo(img, &info);

		if (info.xRes != tracker->GetWidth() || info.yRes != tracker->GetHeight())
			return;

		float median = storedMedian ? *storedMedian : -1.0f;
		
		if (info.imageType == IMAQ_IMAGE_U8) {
			uchar* imgData = (uchar*)info.imageStart;
			tracker->SetImage8Bit(imgData, info.xRes, info.yRes, info.pixelsPerLine);
		} else if(info.imageType == IMAQ_IMAGE_U16) {
			ushort* imgData = (ushort*)info.imageStart;
			tracker->SetImage16Bit(imgData, info.xRes, info.yRes, info.pixelsPerLine*2);
		} else
			return;

		if (median < 0.0f) median = tracker->ComputeMedian();
		vector2f com = tracker->ComputeCOM(median);

		COM[0] = com.x;
		COM[1] = com.y;
				
		vector2f xcorpos = tracker->ComputeXCorInterpolated(com, xcor_iterations);
		xcor[0] = xcorpos.x;
		xcor[1] = xcorpos.y;

		if (dbgImg) {
#ifdef _DEBUG
			ImageInfo di;
			imaqGetImageInfo(dbgImg, &di);
			float* img = tracker->GetDebugImage();
			if (di.imageType == IMAQ_IMAGE_U16 && dbgImg) {
				ushort* d = floatToNormalizedUShort(img, tracker->GetWidth(), tracker->GetHeight());
				imaqArrayToImage(dbgImg, d, info.xRes, info.yRes);
				delete[] d;
			}
#else
			ImageInfo di;
			imaqGetImageInfo(dbgImg, &di);
			if (di.imageType == IMAQ_IMAGE_U16 && tracker->GetDebugImage()) {
				ushort* d = floatToNormalizedUShort(tracker->GetDebugImage(), tracker->GetWidth(), tracker->GetHeight());
				imaqArrayToImage(dbgImg, d, info.xRes, info.yRes);
				delete[] d;
			}
#endif
		}
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
	}
}

/*

median = 0: Use zero median
*median < 0: Compute median, use it and store
*median >= 0: Use given median

*/
CDLL_EXPORT void DLL_CALLCONV compute_com(Tracker* tracker, float *median, float* out)
{
	float m;
	if (!median)
		m = 0.0f;
	else if (median) {
		if (*median >= 0.0f)
			m = *median;
		else {
			m = tracker->ComputeMedian();
			*median = m;
		}
	}
	vector2f com = tracker->ComputeCOM(m);
	out[0] = com.x;
	out[1] = com.y;
}

CDLL_EXPORT void DLL_CALLCONV compute_xcor(Tracker* tracker, vector2f* position, int iterations)
{
	*position = tracker->ComputeXCorInterpolated(*position,iterations);
}

CDLL_EXPORT void DLL_CALLCONV set_image(Tracker* tracker, Image* img, int offsetX, int offsetY)
{
	try {
		ImageInfo info;
		imaqGetImageInfo(img, &info);

		if (offsetX < 0 || offsetY < 0 || offsetX + tracker->GetWidth() > info.xRes || offsetY + tracker->GetHeight() > info.yRes)
			return;
		if (info.imageType == IMAQ_IMAGE_U8)
			tracker->SetImage8Bit((uchar*)info.imageStart + offsetX + info.pixelsPerLine * offsetY, info.xRes, info.yRes, info.pixelsPerLine);
		else if(info.imageType == IMAQ_IMAGE_U16)
			tracker->SetImage16Bit((ushort*)info.imageStart + offsetX + info.pixelsPerLine * offsetY, info.xRes, info.yRes, info.pixelsPerLine*2);
		else
			return;
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
	}
}

CDLL_EXPORT void DLL_CALLCONV compute_radial_profile(Tracker* tracker, ppFloatArray result, int angularSteps, float range, float* center)
{
	LVFloatArray* dst = *result;
	tracker->ComputeRadialProfile(&dst->elt[0], dst->dimSize, angularSteps, range, *(vector2f*)center);
}

CDLL_EXPORT void DLL_CALLCONV set_ZLUT(Tracker* tracker, ppFloatArray2 pZlut, float profile_radius)
{
	LVFloatArray2* zlut = *pZlut;
	
	tracker->SetZLUT(zlut->data, zlut->dimSizes[0], zlut->dimSizes[1], profile_radius);
}

CDLL_EXPORT float DLL_CALLCONV compute_z(Tracker* tracker, float* center, int angularSteps)
{
	return tracker->ComputeZ(*(vector2f*)center, angularSteps);
}

CDLL_EXPORT void DLL_CALLCONV get_debug_image(Tracker* tracker, Image* dbgImg)
{
	ImageInfo di;
	imaqGetImageInfo(dbgImg, &di);
	if (di.imageType == IMAQ_IMAGE_U16) {
		float* dbg = tracker->GetDebugImage();
		if (dbg) {
			ushort* d = floatToNormalizedUShort(dbg, tracker->GetWidth(), tracker->GetHeight());
			imaqArrayToImage(dbgImg, d, tracker->GetWidth(), tracker->GetHeight());
			delete[] d;
		}
	}
}

CDLL_EXPORT TrackerQueue* create_queue(int workerThreads, int width, int height, int xcorw, ppFloatArray2 pZlut, float profile_radius)
{
	LVFloatArray2* zlutdata = *pZlut;
	ZLookupTable* zlut = new ZLookupTable (zlutdata->data, zlutdata->dimSizes[0], zlutdata->dimSizes[1], profile_radius);

	// zlut is now owned by TrackerQueue
	return 0;
}

