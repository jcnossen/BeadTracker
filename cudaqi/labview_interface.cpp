#include "nivision.h"
#include "../cudaqi/cudaqi.h"
#include "extcode.h"
#include "niimaq.h"

#include <Windows.h>
#include "utils.h"
#include "../cudaqi/GPUBase.h"
#include "../cudaqi/GPUImage.h"

#define CALLCONV _FUNCC

#pragma pack(push, 4)
struct vector2f {
	float x,y;
};
#pragma pack(pop)


std::string SPrintf(const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);

	char buf[512];
	VSNPRINTF(buf, sizeof(buf), fmt, ap);

	va_end(ap);
	return buf;
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


DLL_EXPORT GPUImage* CALLCONV copy_to_gpu(Image* img)
{
	ImageInfo info;
	imaqGetImageInfo(img, &info);

	GPUImage* gpuImg = GPUImage::buildFrom8bitStrided((uint8_t*)info.imageStart, info.pixelsPerLine, info.xRes, info.yRes);
	return gpuImg;
}


DLL_EXPORT void CALLCONV free_gpu_image(GPUImage* img)
{
	delete img;
}

DLL_EXPORT void CALLCONV xcor_localize(GPUImage* img, const vector2f& initial, vector2f& position)
{

}

