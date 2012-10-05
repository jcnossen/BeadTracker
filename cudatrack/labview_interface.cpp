#include "nivision.h"
#include "extcode.h"
#include "niimaq.h"

#include <Windows.h>
#include "utils.h"
#include "GPUBase.h"

#include "tracker.h"

#define CALLCONV _FUNCC

/* lv_prolog.h and lv_epilog.h set up the correct alignment for LabVIEW data. */
#include "lv_prolog.h"
struct LVFloatArray {
	int32_t dimSize;
	float elt[1];
};
typedef LVFloatArray **TD1Hdl;
#include "lv_epilog.h"



ushort* floatToNormalizedUShort(float *data, uint w,uint h)
{
	float maxv = data[0];
	float minv = data[0];
	for (uint k=0;k<w*h;k++) {
		maxv = max(maxv, data[k]);
		minv = min(minv, data[k]);
	}
	ushort *norm = new ushort[w*h];
	for (uint k=0;k<w*h;k++)
		norm[k] = ((1<<16)-1) * (data[k]-minv) / (maxv-minv);
	return norm;
}

DLL_EXPORT void CALLCONV generate_test_image(Image *img, uint w, uint h, float xp, float yp, float size)
{
	float S = 1.0f/size;
	float *d = new float[w*h];
	for (uint y=0;y<h;y++)
		for (uint x=0;x<w;x++) {
			float X = x - xp;
			float Y = y - yp;
			float r = sqrtf(X*X+Y*Y)+1;
			float v = 0.1 + sinf( (r-10)/5) * expf(-r*S);
			d[y*w+x] = v;
		}

	ushort* result = floatToNormalizedUShort(d, w, h);
	imaqArrayToImage(img, result, w,h);
	delete[] result;
	delete[] d;
}



void copyToLVArray (TD1Hdl r, const std::vector<float>& a)
{
	LVFloatArray* dst = *r;

	int len = min( dst->dimSize, a.size () );
//	dbgout(SPrintf("copying %d elements to Labview array\n", len));
	for (uint i=0;i<a.size();i++)
		dst->elt[i] = a[i];
}


DLL_EXPORT void CALLCONV copy_crosscorrelation_result(Tracker* tracker, TD1Hdl x_result, TD1Hdl y_result, TD1Hdl x_xc, TD1Hdl y_xc)
{
}

DLL_EXPORT void CALLCONV localize_image(Tracker* tracker, Image* img, float* COM, float* xcor,  float* median, Image* dbgImg, int xcor_iterations)
{
}

DLL_EXPORT Tracker* CALLCONV create_tracker(int w,int h)
{
	return new Tracker(w,h);
}

DLL_EXPORT void CALLCONV free_tracker(Tracker* tracker)
{
	if (tracker->isValid()) {
		delete tracker;
	}
}


DLL_EXPORT Image* CALLCONV copy_image(Image* image)
{
	ImageInfo info, dstInfo;
	imaqGetImageInfo(image, &info);

	Image* output = imaqCreateImage(IMAQ_IMAGE_U8, 0);
	imaqSetImageSize(output, info.xRes, info.yRes);
	imaqGetImageInfo(output, &dstInfo);

	uint8_t *src = (uint8_t*)imaqGetPixelAddress(image, Point());
	for (int y=0;y<info.yRes;y++) {
		src += info.pixelsPerLine;

		for (int x=0;x<info.xRes;x++) {
			Point pt = {x,y};
			PixelValue v;
			v.grayscale = 255-src[x];
			imaqSetPixel(output, pt, v);
		}
	}

	return output;
}

DLL_EXPORT void CALLCONV compute_com(Tracker* tracker, float* pos)
{
	vector2f com = tracker->computeCOM();
	pos[0] = com.x;
	pos[1] = com.y;
}




DLL_EXPORT void CALLCONV testMsg() 
{
	MessageBox(0, "hi", "Msg:", MB_OK);
}


DLL_EXPORT void CALLCONV xcor_localize(Tracker* tracker, const vector2f& initial, vector2f& position)
{

}

DLL_EXPORT void CALLCONV get_current_image(Tracker* tracker, Image* target)
{
	pixel_t *data = new pixel_t[tracker->width*tracker->height];
	tracker->copyToHost(data, tracker->width*sizeof(pixel_t));

	int width, height;
	imaqGetImageSize(target, &width, &height);
	if (width != tracker->width || height != tracker->height)
		imaqSetImageSize(target, tracker->width, tracker->height);

	imaqArrayToImage(target, data, tracker->width, tracker->height);
	delete[] data;
}

DLL_EXPORT void CALLCONV set_image(Tracker* tracker, Image* img)
{
	ImageInfo info;
	imaqGetImageInfo(img, &info);
	if (info.imageType == IMAQ_IMAGE_U8 && info.xRes == tracker->width && info.yRes == tracker->height)
		tracker->setImage( (uchar*)info.imageStart, info.pixelsPerLine);

}
