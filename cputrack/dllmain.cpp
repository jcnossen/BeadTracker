#include <Windows.h>
#undef max
#undef min

//#include "extcode.h"
#include "labview.h"




int __stdcall DllMain (void *hinstDLL, int fdwReason, void *lpvReserved)
{
	if (fdwReason == DLL_PROCESS_ATTACH)
	{
	}
	else if (fdwReason == DLL_PROCESS_DETACH)
	{
	}
	return 1;
}

extern "C" __declspec(dllexport) void* do_stuff(int x)
{
	return 0;
}


extern "C" __declspec(dllexport) void free_stuff(void *d)
{
	
}


