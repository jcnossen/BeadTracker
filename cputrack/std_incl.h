#pragma once

#ifdef USE_MEMDBG
// CRT memory leak debugging is not straightforward to get working. Read comments in 
// http://msdn.microsoft.com/en-us/library/e5ewb1h3(v=vs.80).aspx for details.
//	#define _CRTDBG_MAP_ALLOC needs to be defined in preprocessor options for some reason?

/*	#include <stdlib.h>
	#include <crtdbg.h>

	#ifndef DEBUG_NEW
	#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
	#define new DEBUG_NEW
	#endif
	*/

	#define new new(__FILE__, __LINE__)

	void* operator new(size_t s, const char* file, int line);
	void operator delete(void* p);
	void operator delete[](void* p);

	void MemDbgListAllocations();

#endif

#pragma pack(push, 4)
struct vector2f {
	float x,y;
};

struct vector3f {
	float x,y,z;
};
#pragma pack(pop)


#define _CRT_SECURE_NO_WARNINGS

#ifdef _MSC_VER
#pragma warning(disable: 4244) // conversion from 'int' to 'float', possible loss of data
#endif


typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned long ulong;
typedef unsigned char uchar;

/*
 * Portable definition for SNPRINTF, VSNPRINTF, STRCASECMP and STRNCASECMP
 */
#ifdef _MSC_VER
	#if _MSC_VER > 1310
		#define SNPRINTF _snprintf_s
		#define VSNPRINTF _vsnprintf_s
	#else
		#define SNPRINTF _snprintf
		#define VSNPRINTF _vsnprintf
	#endif
	#define STRCASECMP _stricmp
	#define STRNCASECMP _strnicmp
	#define ALLOCA(size) _alloca(size) // allocates memory on stack
#else
	#define STRCASECMP strcasecmp
	#define STRNCASECMP strncasecmp
	#define SNPRINTF snprintf
	#define VSNPRINTF vsnprintf
	#define ALLOCA(size) alloca(size)
#endif
#define ALLOCA_ARRAY(T, N) ((T*)ALLOCA(sizeof(T) * N))

#define DLL_CALLCONV __cdecl
#define CDLL_EXPORT extern "C" __declspec(dllexport) 
#define DLL_EXPORT __declspec(dllexport) 
