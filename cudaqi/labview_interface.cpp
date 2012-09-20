#include "nivision.h"
#include "extcode.h"
#include "niimaq.h"

#include <Windows.h>
#include "utils.h"
#include "GPUBase.h"

#include "tracker.h"

#define CALLCONV _FUNCC

static std::string lastErrorMsg;


std::string SPrintf(const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);

	char buf[512];
	VSNPRINTF(buf, sizeof(buf), fmt, ap);

	va_end(ap);
	return buf;
}


DLL_EXPORT Tracker* CALLCONV create_tracker(int w,int h)
{
	return new Tracker(w,h);
}

DLL_EXPORT void CALLCONV free_tracker(Tracker* tracker)
{
	if (tracker->isValid())
		delete tracker;
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

DLL_EXPORT void CALLCONV compute_com(GPUImage* img, float* pos)
{
/*	ImageInfo info;
	imaqGetImageInfo(img, &info);

	void* p = imaqGetPixelAddress(img, Point());

	if (info.imageType == IMAQ_IMAGE_U8) {
		uint8_t *data = (uint8_t*)p;

		for (int y=0;y<info.yRes;y++) {
			for (int x=0;x<info.xRes;x++) {
			}
		}
	}*/
}




DLL_EXPORT void CALLCONV testMsg() 
{
	MessageBox(0, "hi", "Msg:", MB_OK);
}




DLL_EXPORT void CALLCONV xcor_localize(Tracker* tracker, const vector2f& initial, vector2f& position)
{

}

