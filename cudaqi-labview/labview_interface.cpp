#include "nivision.h"
#include "../cudaqi/cudaqi.h"
#include "extcode.h"
#include "niimaq.h"

#include <Windows.h>

#define DLL_EXPORT extern "C" __declspec(dllexport) 

struct Position
{
	float x,y,z;
};


DLL_EXPORT void __cdecl process_image(int32_t *position, uintptr_t *imagePtr)
{
	ImageInfo info;
	Image* image = (Image*)imagePtr;
	imaqGetImageInfo(image, &info);

	*position = info.xRes;
}


DLL_EXPORT void __cdecl testMsg() 
{
	MessageBox(0, "hi", "Msg:", MB_OK);
}

