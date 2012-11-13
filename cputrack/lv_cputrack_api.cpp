/*
Labview API for CPU tracker
*/
#include <Windows.h>

#include "random_distr.h"
#include "labview.h"
#include "cpu_tracker.h"

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


CDLL_EXPORT CPUTracker* DLL_CALLCONV create_tracker(uint w, uint h, uint xcorw)
{
	try {
		Sleep(300);
		return new CPUTracker(w,h,xcorw);
	}
	catch(const std::exception& e)
	{
		dbgout("Exception: " + std::string(e.what()) + "\n");
		return 0;
	}
}

CDLL_EXPORT void DLL_CALLCONV destroy_tracker(CPUTracker* tracker)
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
	for (size_t x=0;x<a.size();x++)
		(*r)->elem[x] = a[x];
}

CDLL_EXPORT void DLL_CALLCONV copy_crosscorrelation_result(CPUTracker* tracker, LVArray<float>** x_result, LVArray<float>** y_result, LVArray<float>** x_xc, LVArray<float>** y_xc)
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

/*

median = 0: Use zero median
*median < 0: Compute median, use it and store
*median >= 0: Use given median

*/
CDLL_EXPORT void DLL_CALLCONV compute_com(CPUTracker* tracker, float* out)
{
	float m;
	vector2f com = tracker->ComputeBgCorrectedCOM();
	out[0] = com.x;
	out[1] = com.y;
}

CDLL_EXPORT void DLL_CALLCONV compute_xcor(CPUTracker* tracker, vector2f* position, int iterations, int profileWidth, int useInterpolation)
{
	if (useInterpolation)
		*position = tracker->ComputeXCorInterpolated(*position, iterations, profileWidth);
	else
		*position = tracker->ComputeXCor(*position, profileWidth);
}


CDLL_EXPORT void DLL_CALLCONV compute_qi(CPUTracker* tracker, vector2f* position, int iterations, int radialSteps, int angularStepsPerQ, float minRadius, float maxRadius)
{
	*position = tracker->ComputeQI(*position, iterations, radialSteps, angularStepsPerQ, minRadius,maxRadius);
}


CDLL_EXPORT void DLL_CALLCONV set_image(CPUTracker* tracker, Image* img, int offsetX, int offsetY, ErrorCluster* error)
{
	try {
		ImageInfo info;
		imaqGetImageInfo(img, &info);

		if (offsetX < 0 || offsetY < 0 || offsetX + tracker->GetWidth() > info.xRes || offsetY + tracker->GetHeight() > info.yRes || info.xRes != tracker->GetWidth() || info.yRes != tracker->GetHeight()) {
			ArgumentErrorMsg(error, "Invalid image dimension or offset given");
			return;
		}
		if (info.imageType == IMAQ_IMAGE_U8)
			tracker->SetImage8Bit((uchar*)info.imageStart + offsetX + info.pixelsPerLine * offsetY, info.pixelsPerLine);
		else if(info.imageType == IMAQ_IMAGE_U16)
			tracker->SetImage16Bit((ushort*)info.imageStart + offsetX + info.pixelsPerLine * offsetY, info.pixelsPerLine*2);
		else
			return;
	}
	catch(const std::exception& e)
	{
		ArgumentErrorMsg(error, "Exception: " + std::string(e.what()) + "\n");
	}
}

CDLL_EXPORT void DLL_CALLCONV set_image_u8(CPUTracker* tracker, LVArray2D<uchar>** pData, ErrorCluster* error)
{
	LVArray2D<uchar>* data = *pData;
	if (data->dimSizes[0] != tracker->GetWidth() || data->dimSizes[1] != tracker->GetHeight()) {
		ArgumentErrorMsg(error, "Given image has invalid dimensions");
		return;
	}
	tracker->SetImage8Bit( data->elem, tracker->GetWidth() );
}

CDLL_EXPORT void DLL_CALLCONV set_image_u16(CPUTracker* tracker, LVArray2D<ushort>** pData, ErrorCluster* error)
{
	LVArray2D<ushort>* data = *pData;
	if (data->dimSizes[0] != tracker->GetWidth() || data->dimSizes[1] != tracker->GetHeight()) {
		ArgumentErrorMsg(error, "Given image has invalid dimensions");
		return;
	}
	tracker->SetImage16Bit( data->elem, tracker->GetWidth()*sizeof(ushort) );
}

CDLL_EXPORT void DLL_CALLCONV set_image_float(CPUTracker* tracker, LVArray2D<float>** pData, ErrorCluster* error)
{
	LVArray2D<float>* data = *pData;
	if (data->dimSizes[0] != tracker->GetWidth() || data->dimSizes[1] != tracker->GetHeight()) {
		ArgumentErrorMsg(error, "Given image has invalid dimensions");
		return;
	}
	tracker->SetImageFloat( data->elem );
}



CDLL_EXPORT float DLL_CALLCONV compute_z(CPUTracker* tracker, float* center, int angularSteps, int zlut_index)
{
	return tracker->ComputeZ(*(vector2f*)center, angularSteps, zlut_index);
}

CDLL_EXPORT void DLL_CALLCONV get_debug_image(CPUTracker* tracker, Image* dbgImg)
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

CDLL_EXPORT void DLL_CALLCONV get_debug_img_as_array(CPUTracker* tracker, LVArray2D<float>** pdbgImg)
{
	float* src = tracker->GetDebugImage();
	if (src) {
		ResizeLVArray2D(pdbgImg, tracker->GetHeight(), tracker->GetWidth());
		LVArray2D<float>* dst = *pdbgImg;

	//	dbgout(SPrintf("copying %d elements to Labview array\n", len));
		for (int i=0;i<dst->numElem();i++)
			dst->elem[i] = src[i];
	}
}



CDLL_EXPORT void DLL_CALLCONV compute_radial_profile(CPUTracker* tracker, LVArray<float>** result, int angularSteps, float range, float* center)
{
	LVArray<float>* dst = *result;
	tracker->ComputeRadialProfile(&dst->elem[0], dst->dimSize, angularSteps, 1.0f, range, *(vector2f*)center);
}




CDLL_EXPORT void DLL_CALLCONV set_ZLUT(CPUTracker* tracker, LVArray3D<float>** pZlut, float profile_radius, int angular_steps)
{
	LVArray3D<float>* zlut = *pZlut;

	int numLUTs = zlut->dimSizes[0];
	int planes = zlut->dimSizes[1];
	int res = zlut->dimSizes[2];
	
	tracker->SetZLUT(zlut->elem, planes, res, numLUTs, 1.0f,  profile_radius, angular_steps, true);
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


