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



DLL_EXPORT void CALLCONV process_image(int32_t *position, uintptr_t *imagePtr)
{
	ImageInfo info;
	Image* image = (Image*)imagePtr;
	imaqGetImageInfo(image, &info);

	*position = info.xRes;
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

