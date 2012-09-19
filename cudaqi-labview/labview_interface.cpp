#include "nivision.h"
#include "../cudaqi/cudaqi.h"
#include "extcode.h"
#include "niimaq.h"

#include <Windows.h>
#include "utils.h"

#define CALLCONV _FUNCC

struct Position
{
	float x,y,z;
};



std::string SPrintf(const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);

	char buf[512];
	VSNPRINTF(buf, sizeof(buf), fmt, ap);

	va_end(ap);
	return buf;
}



DLL_EXPORT uintptr_t CALLCONV copy_image(uintptr_t srcPtr)
{
	ImageInfo info, dstInfo;
	Image* image = (Image*)srcPtr;
	imaqGetImageInfo(image, &info);

	Image* output = imaqCreateImage(IMAQ_IMAGE_U8, 0);
	imaqSetImageSize(output, info.xRes, info.yRes);
	imaqGetImageInfo(output, &dstInfo);

	uint8_t *src = (uint8_t*)imaqGetPixelAddress(image, Point());
	uint8_t *dst = (uint8_t*)imaqGetPixelAddress(output, Point());
	for (int y=0;y<info.yRes;y++) {
		src += info.pixelsPerLine;
		dst += info.pixelsPerLine;

		for (int x=0;x<info.xRes;x++) {
			Point pt = {x,y};
			PixelValue v;
			v.grayscale = 255-src[x];
			imaqSetPixel(output, pt, v);
		}
	}

	return (uintptr_t)output;
}

DLL_EXPORT void CALLCONV compute_com(uintptr_t *imagePtr, float* pos)
{
	Image* img = (Image*)imagePtr;
	
	ImageInfo info;
	imaqGetImageInfo(img, &info);

	void* p = imaqGetPixelAddress(img, Point());

	if (info.imageType == IMAQ_IMAGE_U8) {
		uint8_t *data = (uint8_t*)p;

		for (int y=0;y<info.yRes;y++) {
			for (int x=0;x<info.xRes;x++) {
			}
		}
	}
}



DLL_EXPORT uintptr_t * CALLCONV test_image()
{
	Image* img = imaqCreateImage(IMAQ_IMAGE_U8, 0);

//	imaqSet

	return (uintptr_t*)img;
}

DLL_EXPORT void CALLCONV testMsg() 
{
	MessageBox(0, "hi", "Msg:", MB_OK);
}

