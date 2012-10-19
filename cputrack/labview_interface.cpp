
#include <Windows.h>

#include "nivision.h"
#include "extcode.h"
#include "niimaq.h"
#include "random_distr.h"

#define CALLCONV _FUNCC

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

#include "cpu_tracker.h"



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

DLL_EXPORT void CALLCONV generate_test_image(Image *img, uint w, uint h, float xp, float yp, float size, float photoncount)
{
	try {
		float S = 1.0f/size;
		float *d = new float[w*h];
		for (uint y=0;y<h;y++)
			for (uint x=0;x<w;x++) {
				float X = x - xp;
				float Y = y - yp;
				float r = sqrtf(X*X+Y*Y)+1;
				float v = 0.1 - sinf( (r-10)/5) * expf(-r*S);
				d[y*w+x] = v;
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


DLL_EXPORT CPUTracker* CALLCONV create_tracker(uint w, uint h, uint xcorw)
{
	try {
		Sleep(300);
		return new CPUTracker(w,h, xcorw);
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
		return 0;
	}
}

DLL_EXPORT void CALLCONV destroy_tracker(CPUTracker* tracker)
{
	try {
		delete tracker;
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
	}
}

void copyToLVArray (ppFloatArray r, const std::vector<float>& a)
{
	size_t wantedSize = sizeof(float)*a.size();

	NumericArrayResize(9 /* codes for SGL type */, 1, (UHandle*)&r, wantedSize);

	LVFloatArray* dst = *r;
	dst->dimSize = a.size();
	size_t len = min( dst->dimSize, a.size () );
//	dbgout(SPrintf("copying %d elements to Labview array\n", len));
	for (size_t i=0;i<a.size();i++)
		dst->elt[i] = a[i];
}

DLL_EXPORT void CALLCONV copy_crosscorrelation_result(CPUTracker* tracker, ppFloatArray x_result, ppFloatArray y_result, ppFloatArray x_xc, ppFloatArray y_xc)
{
	try {
		if (x_result) copyToLVArray (x_result, tracker->X_result);
		if (y_result) copyToLVArray (y_result, tracker->Y_result);
		if (x_xc) copyToLVArray (x_xc, tracker->X_xc);
		if (y_xc) copyToLVArray (y_xc, tracker->Y_xc);
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
	}
}

DLL_EXPORT void CALLCONV localize_image(CPUTracker* tracker, Image* img, float* COM, float* xcor,  float* storedMedian, Image* dbgImg, int xcor_iterations)
{
	try {
		ImageInfo info;
		imaqGetImageInfo(img, &info);

		if (info.xRes != tracker->width || info.yRes != tracker->height)
			return;

		vector2f com;
		if (info.imageType == IMAQ_IMAGE_U8) {
			uchar* imgData = (uchar*)info.imageStart;

			float median = ComputeMedian(imgData, info.xRes, info.yRes, info.pixelsPerLine, storedMedian);
			tracker->SetImage(imgData, info.xRes, info.yRes, info.pixelsPerLine);
			com = tracker->ComputeCOM(median);
		} else if(info.imageType == IMAQ_IMAGE_U16) {
			ushort* imgData = (ushort*)info.imageStart;
			float median = ComputeMedian(imgData, info.xRes, info.yRes, info.pixelsPerLine*2, storedMedian);
			tracker->SetImage(imgData, info.xRes, info.yRes, info.pixelsPerLine*2);
			com = tracker->ComputeCOM(median);
		} else
			return;

//		COM[0] = com.x;
	//	COM[1] = com.y;

		com.x = info.xRes/2;
		com.y = info.yRes/2;
		
		vector2f xcorpos = tracker->ComputeXCor(com);//tracker->ComputeXCor(com, xcor_iterations);
		xcor[0] = xcorpos.x;
		xcor[1] = xcorpos.y;

		if (dbgImg) {
#ifdef _DEBUG
			ImageInfo di;
			imaqGetImageInfo(dbgImg, &di);
			if (di.imageType == IMAQ_IMAGE_U16) {
				ushort* d = floatToNormalizedUShort(tracker->debugImage, tracker->width, tracker->height);
				imaqArrayToImage(dbgImg, d, info.xRes, info.yRes);
				delete[] d;
			}
#else
			ImageInfo di;
			imaqGetImageInfo(dbgImg, &di);
			if (di.imageType == IMAQ_IMAGE_U16) {
				ushort* d = floatToNormalizedUShort(tracker->srcImage, tracker->width, tracker->height);
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
DLL_EXPORT void CALLCONV compute_com(CPUTracker* tracker, float *median, float* out)
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

DLL_EXPORT void CALLCONV compute_xcor(CPUTracker* tracker, vector2f* position, int iterations)
{
	*position = tracker->ComputeXCor(*position);
}

DLL_EXPORT void CALLCONV set_image(CPUTracker* tracker, Image* img, int offsetX, int offsetY)
{
	try {
		ImageInfo info;
		imaqGetImageInfo(img, &info);

		if (offsetX < 0 || offsetY < 0 || offsetX + tracker->width > info.xRes || offsetY + tracker->height > info.yRes)
			return;
		if (info.imageType == IMAQ_IMAGE_U8)
			tracker->SetImage((uchar*)info.imageStart + offsetX + info.pixelsPerLine * offsetY, tracker->width, tracker->height, info.pixelsPerLine);
		else if(info.imageType == IMAQ_IMAGE_U16)
			tracker->SetImage((ushort*)info.imageStart + offsetX + info.pixelsPerLine * offsetY, info.xRes, info.yRes, info.pixelsPerLine*2);
		else 
			return;
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
	}
}

DLL_EXPORT void CALLCONV compute_radial_profile(CPUTracker* tracker, ppFloatArray result, int angularSteps, float range, float* center)
{
	LVFloatArray* dst = *result;
	tracker->ComputeRadialProfile(&dst->elt[0], dst->dimSize, angularSteps, range, *(vector2f*)center);
}

DLL_EXPORT void CALLCONV set_ZLUT(CPUTracker* tracker, ppFloatArray2 pZlut)
{
	LVFloatArray2* zlut = *pZlut;
	
	tracker->SetZLUT(zlut->data, zlut->dimSizes[0], zlut->dimSizes[1]);
}


DLL_EXPORT void CALLCONV get_debug_image(CPUTracker* tracker, Image* dbgImg)
{
	ImageInfo di;
	imaqGetImageInfo(dbgImg, &di);
	if (di.imageType == IMAQ_IMAGE_U16) {
		ushort* d = floatToNormalizedUShort(tracker->debugImage, tracker->width, tracker->height);
		imaqArrayToImage(dbgImg, d, tracker->width, tracker->height);
		delete[] d;
	}
}


//DLL_EXPORT void CALLCONV compute_xcor_2D(CPUTracker* tracker, vector2f position, 

