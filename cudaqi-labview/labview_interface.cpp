#include "nivision.h"
#include "../cudaqi/cudaqi.h"

#include <Windows.h>


#define DLL_EXPORT extern "C" __declspec(dllexport) 
struct CudaImage;

struct Position
{
	float x,y,z;
};

DLL_EXPORT Position cudaqiQI(CudaImage* image, Position* initial)
{
	return *initial;
}


DLL_EXPORT void testMsg() 
{
	MessageBox(0, "hi", "Msg:", MB_OK);
}

