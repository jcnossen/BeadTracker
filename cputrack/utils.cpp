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

ushort* floatToNormalizedUShort(float *src, uint w,uint h)
{ 
	ushort* r = new ushort[w*h]; 
	floatToNormalizedUShort(r,src,w,h); 
	return r; 
}

void floatToNormalizedUShort(ushort* dst, float *src, uint w,uint h)
{
	float maxv = src[0];
	float minv = src[0];
	for (uint k=0;k<w*h;k++) {
		maxv = max(maxv, src[k]);
		minv = min(minv, src[k]);
	}
	for (uint k=0;k<w*h;k++)
		dst[k] = ((1<<16)-1) * (src[k]-minv) / (maxv-minv);
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

void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius,
	vector2f center, float* srcImage, int width, int height)
{
	vector2f* radialDirs = (vector2f*)ALLOCA(sizeof(vector2f)*angularSteps);
	for (int j=0;j<angularSteps;j++) {
		float ang = 2*3.141593f*j/(float)angularSteps;
		vector2f d = { cosf(ang), sinf(ang) };
		radialDirs[j] = d;
	}

	for (int i=0;i<radialSteps;i++)
		dst[i]=0.0f;

	float total = 0.0f;
	float rstep = (maxradius-minradius) / radialSteps;
	for (int i=0;i<radialSteps; i++) {
		float sum = 0.0f;

		float r = minradius+rstep*i;
		for (int a=0;a<angularSteps;a++) {
			float x = center.x + radialDirs[a].x * r;
			float y = center.y + radialDirs[a].y * r;
			sum += Interpolate(srcImage,width,height, x,y);
		}

		dst[i] = sum;
		total += dst[i];
	}
	for (int i=0;i<radialSteps;i++)
		dst[i] /= total;
}
