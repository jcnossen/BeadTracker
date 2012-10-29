
#include <Windows.h>

#include "random_distr.h"
#include "labview.h"
#include "tracker.h"
#include "TrackerQueue.h"

static MgErr FillErrorCluster(MgErr err, const char *message, ErrorCluster *error)
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
void copyToLVArray (LVArray<T>**& r, const std::vector<T>& a)
{
	ResizeLVArray(r, a.size());
	for (int x=0;x<a.size();x++)
		(*r)->elt[x] = a[x];
}

CDLL_EXPORT void DLL_CALLCONV copy_crosscorrelation_result(Tracker* tracker, LVArray<float>** x_result, LVArray<float>** y_result, LVArray<float>** x_xc, LVArray<float>** y_xc)
{
	try {
		std::vector<xcor_t> xprof, yprof, xconv, yconv;
		if (tracker->GetLastXCorProfiles(xprof, yprof, xconv, yconv)) {
			if (x_result) copyToLVArray (x_result, xconv);
			if (y_result) copyToLVArray (y_result, yconv);
			if (x_xc) copyToLVArray (x_xc, xprof);
			if (y_xc) copyToLVArray (y_xc, yprof);
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


CDLL_EXPORT void DLL_CALLCONV set_image(Tracker* tracker, Image* img, int offsetX, int offsetY, ErrorCluster* error)
{
	try {
		ImageInfo info;
		imaqGetImageInfo(img, &info);

		if (offsetX < 0 || offsetY < 0 || offsetX + tracker->GetWidth() > info.xRes || offsetY + tracker->GetHeight() > info.yRes) {
			ArgumentErrorMsg(error, "Invalid image dimension or offset given");
			return;
		}
		if (info.imageType == IMAQ_IMAGE_U8)
			tracker->SetImage8Bit((uchar*)info.imageStart + offsetX + info.pixelsPerLine * offsetY, info.xRes, info.yRes, info.pixelsPerLine);
		else if(info.imageType == IMAQ_IMAGE_U16)
			tracker->SetImage16Bit((ushort*)info.imageStart + offsetX + info.pixelsPerLine * offsetY, info.xRes, info.yRes, info.pixelsPerLine*2);
		else
			return;
	}
	catch(const std::exception& e)
	{
		ArgumentErrorMsg(error, "Exception: " + std::string(e.what()) + "\n");
	}
}

CDLL_EXPORT void DLL_CALLCONV set_image_as_byte_array(Tracker* tracker, LVArray2D<uchar>** data)
{
	tracker->SetImage8Bit( (*data)->data, tracker->GetWidth(), tracker->GetHeight(), tracker->GetWidth() );
}

CDLL_EXPORT void DLL_CALLCONV set_image_as_float_array(Tracker* tracker, LVArray2D<float>** data)
{
	tracker->SetImageFloat( (*data)->data );
}


CDLL_EXPORT void DLL_CALLCONV compute_radial_profile(Tracker* tracker, ppFloatArray result, int angularSteps, float range, float* center)
{
	LVArray<float>* dst = *result;
	tracker->ComputeRadialProfile(&dst->elt[0], dst->dimSize, angularSteps, range, *(vector2f*)center);
}

CDLL_EXPORT void DLL_CALLCONV set_ZLUT(Tracker* tracker, LVArray3D<float>** pZlut, float profile_radius)
{
	LVArray3D<float>* zlut = *pZlut;

	int numLUTs = zlut->dimSizes[0];
	int planes = zlut->dimSizes[1];
	int res = zlut->dimSizes[2];
	
	tracker->SetZLUT(zlut->data, planes, res, numLUTs, profile_radius);
}

CDLL_EXPORT float DLL_CALLCONV compute_z(Tracker* tracker, float* center, int angularSteps, int zlut_index)
{
	return tracker->ComputeZ(*(vector2f*)center, angularSteps, zlut_index);
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

CDLL_EXPORT void DLL_CALLCONV get_debug_img_as_array(Tracker* tracker, ppFloatArray2 pdbgImg)
{
	float* src = tracker->GetDebugImage();
	if (src) {
		ResizeLVArray2D(pdbgImg, tracker->GetHeight(), tracker->GetWidth());
		LVArray2D<float>* dst = *pdbgImg;

	//	dbgout(SPrintf("copying %d elements to Labview array\n", len));
		for (size_t i=0;i<dst->numElem();i++)
			dst->data[i] = src[i];
	}
}

CDLL_EXPORT TrackerQueue* create_queue(int workerThreads, int width, int height, int xcorw, ppFloatArray2 pZlut, float profile_radius)
{
	LVArray2D<float>* zlutdata = *pZlut;
	ZLookupTable* zlut = new ZLookupTable (zlutdata->data, zlutdata->dimSizes[0], zlutdata->dimSizes[1], profile_radius);

	// zlut is now owned by TrackerQueue
	return 0;
}


