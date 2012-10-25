#include <cstdarg>
#include "utils.h"
#include <Windows.h>
#include <string>

#include "random_distr.h"

std::string SPrintf(const char *fmt, ...) {
	va_list ap;
	va_start(ap, fmt);

	char buf[512];
	VSNPRINTF(buf, sizeof(buf), fmt, ap);

	va_end(ap);
	return buf;
}

void dbgout(std::string s) {
	OutputDebugString(s.c_str());
	printf(s.c_str());
}

void dbgprintf(const char *fmt,...) {
	va_list ap;
	va_start(ap, fmt);

	char buf[512];
	VSNPRINTF(buf, sizeof(buf), fmt, ap);
	OutputDebugString(buf);
	puts(buf);

	va_end(ap);
}

ushort* floatToNormalizedUShort(float *data, uint w,uint h)
{
	float maxv = data[0];
	float minv = data[0];
	for (uint k=0;k<w*h;k++) {
		maxv = max(maxv, data[k]);
		minv = min(minv, data[k]);
	}
	ushort *norm = new ushort[w*h];
	for (uint k=0;k<w*h;k++)
		norm[k] = ((1<<16)-1) * (data[k]-minv) / (maxv-minv);
	return norm;
}



void GenerateTestImage(float* data, int w, int h, float xp, float yp, float size, float MaxPhotons)
{
	float S = 1.0f/size;
	for (int y=0;y<h;y++) {
		for (int x=0;x<w;x++) {
			float X = x - xp;
			float Y = y - yp;
			float r = sqrtf(X*X+Y*Y)+1;
			float v = sinf(r/(5*S)) * expf(-r*r*S*0.01f);
			data[y*w+x] = v;
		}
	}

	if (MaxPhotons>0) {
		normalize(data,w,h);
		for (int k=0;k<w*h;k++) {
			data[k] = rand_poisson(data[k]*MaxPhotons);
		}
	}
	normalize(data,w,h);
}

