#pragma once

#include <crtdbg.h>
#include "memdbg.h"
#include <cstdint>
#include <string>
#include <deque>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <complex>



#pragma pack(push, 4)
struct vector2f {
	vector2f() {x=y=0.0f; }
	vector2f(float X,float Y) { x=X;y=Y; }
	float x,y;
};

struct vector3f {
	vector3f() { x=y=z=0.0f; }
	vector3f(float X,float Y,float Z) { x=X; y=Y; z=Z; }
	float x,y,z;

	vector3f operator*(const vector3f& o) const {
		return vector3f(x*o.x,y*o.y,z*o.z);
	}
	vector3f operator+(const vector3f& o) const {
		return vector3f(x+o.x,y+o.y,z+o.z);
	}
	vector3f operator-(const vector3f& o) const {
		return vector3f(x-o.x,y-o.y,z-o.z);
	}
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

#include "dllmacros.h"
