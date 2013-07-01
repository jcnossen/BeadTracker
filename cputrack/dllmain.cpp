#include "std_incl.h"
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


