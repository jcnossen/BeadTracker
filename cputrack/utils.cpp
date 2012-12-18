#include <cstdarg>
#include "utils.h"
#include <Windows.h>
#undef min
#undef max
#include <string>

#include "random_distr.h"
#include "LsqQuadraticFit.h"

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
		maxv = std::max(maxv, src[k]);
		minv = std::min(minv, src[k]);
	}
	for (uint k=0;k<w*h;k++)
		dst[k] = ((1<<16)-1) * (src[k]-minv) / (maxv-minv);
}



void GenerateTestImage(ImageData& img, float xp, float yp, float size, float MaxPhotons)
{
	float S = 1.0f/size;
	for (int y=0;y<img.h;y++) {
		for (int x=0;x<img.w;x++) {
			float X = x - xp;
			float Y = y - yp;
			float r = sqrtf(X*X+Y*Y)+1;
			float v = sinf(r/(5*S)) * expf(-r*r*S*0.01f);
			img.at(x,y)=v;
		}
	}

	if (MaxPhotons>0) {
		normalize(img.data,img.w,img.h);
		for (int k=0;k<img.numPixels();k++) {
			img.data[k] = rand_poisson(img.data[k]*MaxPhotons);
		}
	}
	normalize(img.data,img.w,img.h);
}

void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius,
	vector2f center, ImageData* img, float* radialweights,float mean)
{
	vector2f* radialDirs = (vector2f*)ALLOCA(sizeof(vector2f)*angularSteps);
	for (int j=0;j<angularSteps;j++) {
		float ang = 2*3.141593f*j/(float)angularSteps;
		vector2f d = { cosf(ang), sinf(ang) };
		radialDirs[j] = d;
	}

	for (int i=0;i<radialSteps;i++)
		dst[i]=0.0f;

	double totalrmssum2 = 0.0f;
	float rstep = (maxradius-minradius) / radialSteps;
	for (int i=0;i<radialSteps; i++) {
		double sum = 0.0f;

		float r = minradius+rstep*i;
		for (int a=0;a<angularSteps;a++) {
			float x = center.x + radialDirs[a].x * r;
			float y = center.y + radialDirs[a].y * r;
			sum += img->interpolate(x,y, mean);
		}

		dst[i] = sum/angularSteps-mean;
		totalrmssum2 += dst[i]*dst[i];
	}
	double invTotalrms = 1.0f/sqrt(totalrmssum2/radialSteps);
	for (int i=0;i<radialSteps;i++) {
		dst[i] *= invTotalrms;
		if (radialweights) dst[i] *= radialweights[i];
	}
}


inline float sq(float x) { return x*x; }

void GenerateImageFromLUT(ImageData* image, ImageData* zlut, float zlut_radius, vector2f pos, float z, float M)
{
	// Generate the interpolated ZLUT 
	float* zinterp; 

	// The two Z planes to interpolate between
	int iz = (int)z;
	if (iz < 0) 
		zinterp = zlut->data;
	else if (iz>=zlut->h-1)
		zinterp = &zlut->data[ (zlut->h-1)*zlut->w ];
	else {
		float* zlut0 = &zlut->data [ (int)z * zlut->w ]; 
		float* zlut1 = &zlut->data [ ((int)z + 1) * zlut->w ];
		zinterp = (float*)ALLOCA(sizeof(float)*zlut->w);
		for (int r=0;r<zlut->w;r++) 
			zinterp[r] = interp(zlut0[r], zlut1[r], z-iz);
	}

	const int len=5;
	float xval[len];
	for (int i=0;i<len;i++)
		xval[i]=i-len/2;

	// Generate the image from the interpolated ZLUT
	for (int y=0;y<image->h;y++)
		for (int x=0;x<image->w;x++) 
		{
			float r = zlut->w * sqrtf( sq(x-pos.x) + sq(y-pos.y) )/(zlut_radius*M);

			if (r > zlut->w-1)
				r = zlut->w-1;

			int minR = std::max(0, std::min( (int)r, zlut->w-len ) );
			LsqSqQuadFit<float> lsq(len, xval, &zinterp[minR]);

			float v = lsq.compute(r-(minR+len/2));
			image->at(x,y) = v; // lsq.compute(r-len/2);
		}
}


void ApplyPoissonNoise(ImageData& img, float factor)
{
	for (int k=0;k<img.numPixels();k++)
		img.data[k] = rand_poisson<float>(factor*img.data[k]);
}




void WriteImageAsCSV(const char* file, float* d, int w,int h)
{
	FILE* f = fopen(file, "w");

	for (int y=0;y<h;y++) {
		for (int x=0;x<w;x++)
		{
			fprintf(f, "%f", d[y*w+x]);
			if(x<w-1) fputs("\t", f); 
		}
		fprintf(f, "\n");
	}

	fclose(f);
}


std::vector<uchar> ReadToByteBuffer(const char *filename)
{
	FILE *f = fopen(filename, "rb");

	fseek(f, 0, SEEK_END);
	int len = ftell(f);
	fseek(f, 0, SEEK_SET);

	std::vector<uchar> buf(len);
	fread(&buf[0], 1,len, f);

	fclose(f);
	return buf;
}
