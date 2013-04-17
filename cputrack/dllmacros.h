#pragma once

#define DLL_CALLCONV __cdecl
#ifdef QTRK_EXPORTS
	#define DLL_EXPORT __declspec(dllexport) 
#else
	#define DLL_EXPORT __declspec(dllimport)
#endif
#define CDLL_EXPORT extern "C" DLL_EXPORT

