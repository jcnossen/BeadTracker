#include "std_incl.h"
#include <cstdarg>
#include "utils.h"
#include <Windows.h>
#undef min
#undef max
#include <string>
#include <complex>

#include "random_distr.h"
#include "LsqQuadraticFit.h"
#include "QueuedTracker.h"

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

void GenerateTestImage(ImageData img, float xp, float yp, float size, float MaxPhotons)
{
	float S = 1.0f/size;
	for (int y=0;y<img.h;y++) {
		for (int x=0;x<img.w;x++) {
			float X = x - xp;
			float Y = y - yp;
			float r = sqrtf(X*X+Y*Y)+1;
			float v = sinf(r/(5*S)) * expf(-r*r*S*0.001f);
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



void ComputeCRP(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius,
	vector2f center, ImageData* img, float paddingValue, float*crpmap)
{
	vector2f* radialDirs = (vector2f*)ALLOCA(sizeof(vector2f)*angularSteps);
	for (int j=0;j<angularSteps;j++) {
		float ang = 2*3.141593f*j/(float)angularSteps;
		vector2f d = { cosf(ang), sinf(ang) };
		radialDirs[j] = d;
	}

	for (int i=0;i<radialSteps;i++)
		dst[i]=0.0f;

	float* map = crpmap ? crpmap : (float*)ALLOCA(sizeof(float)*radialSteps*angularSteps);
	float* com = (float*)ALLOCA(sizeof(float)*angularSteps);

	float rstep = (maxradius-minradius) / radialSteps;
	float comsum = 0.0f;
	for (int a=0;a<angularSteps;a++) {
		float r = minradius;
		float sum = 0.0f, moment=0.0f;
		for (int i=0;i<radialSteps; i++) {
			float x = center.x + radialDirs[a].x * r;
			float y = center.y + radialDirs[a].y * r;
			float v = img->interpolate(x,y, paddingValue);
			r += rstep;
			map[a*radialSteps+i] = v;
			sum += v;
			moment += i*v;
		}
		com[a] = moment/sum;
		comsum += com[a];
	}
	float avgcom = comsum/angularSteps;
	float totalrmssum2 = 0.0f;
	for (int i=0;i<radialSteps; i++) {
		double sum = 0.0f;
		for (int a=0;a<angularSteps;a++) {
			float shift = com[a]-avgcom;
			sum += map[a*radialSteps+i];
		}
		dst[i] = sum/angularSteps-paddingValue;
		totalrmssum2 += dst[i]*dst[i];
	}
	double invTotalrms = 1.0f/sqrt(totalrmssum2/radialSteps);
	for (int i=0;i<radialSteps;i++) {
		dst[i] *= invTotalrms;
	}
}


float ComputeBgCorrectedCOM1D(float *data, int len, float cf)
{
	float sum=0, sum2=0;
	float moment=0;

	for (int x=0;x<len;x++) {
		float v = data[x];
		sum += v;
		sum2 += v*v;
	}

	float invN = 1.0f/len;
	float mean = sum * invN;
	float stdev = sqrtf(sum2 * invN - mean * mean);
	sum = 0.0f;

	for(int x=0;x<len;x++)
	{
		float v = data[x];
		v = std::max(0.0f, fabs(v-mean)-cf*stdev);
		sum += v;
		moment += x*v;
	}
	return moment / (float)sum;
}

void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius,
	vector2f center, ImageData* img, float paddingValue)
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
			sum += img->interpolate(x,y, paddingValue);
		}

		dst[i] = sum/angularSteps;
	}
		/*
		dst[i] = sum/angularSteps-paddingValue;
		totalrmssum2 += dst[i]*dst[i];
	}
	double invTotalrms = 1.0f/sqrt(totalrmssum2/radialSteps);
	for (int i=0;i<radialSteps;i++) {
		dst[i] *= invTotalrms;
	}*/
}


inline float sq(float x) { return x*x; }

void GenerateImageFromLUT(ImageData* image, ImageData* zlut, float lutminRadius, float lutmaxRadius, vector2f pos, float z, float M)
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
			float pixr = sqrtf( sq(x-pos.x) + sq(y-pos.y) );
			float r = zlut->w * ( pixr - lutminRadius ) / ( (lutmaxRadius - lutminRadius) * M);

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

void ApplyGaussianNoise(ImageData& img, float sigma)
{
	for (int k=0;k<img.numPixels();k++) {
		float v = img.data[k] + sigma * rand_normal<float>();
		if (v<0.0f) v= 0.0f;
		img.data[k]=v;
	}
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


void WriteComplexImageAsCSV(const char* file, std::complex<float>* d, int w,int h)
{
	FILE* f = fopen(file, "w");

	for (int y=0;y<h;y++) {
		for (int x=0;x<w;x++)
		{
			fprintf(f, "%f+%fi", d[y*w+x].real(), d[y*w+x].imag());
			if(x<w-1) fputs("\t", f); 
		}
		fprintf(f, "\n");
	}

	fclose(f);
}

std::vector<uchar> ReadToByteBuffer(const char *filename)
{
	FILE *f = fopen(filename, "rb");

	if (!f)
		throw std::runtime_error(SPrintf("%s was not found", filename));

	fseek(f, 0, SEEK_END);
	int len = ftell(f);
	fseek(f, 0, SEEK_SET);

	std::vector<uchar> buf(len);
	fread(&buf[0], 1,len, f);

	fclose(f);
	return buf;
}

void CopyImageToFloat(uchar* data, int width, int height, int pitch, QTRK_PixelDataType pdt, float* dst)
{
	if (pdt == QTrkU8) {
		for (int y=0;y<height;y++) {
			for (int x=0;x<width;x++)
				dst[x] = data[x];
			data += pitch;
			dst += width;
		}
	} else if(pdt == QTrkU16) {
		for (int y=0;y<height;y++) {
			ushort* u = (ushort*)data;
			for (int x=0;x<width;x++)
				dst[x] = u[x];
			data += pitch;
			dst += width;
		}
 	} else {
		for (int y=0;y<height;y++) {
			memcpy(dst, (float*)data, sizeof(float)*width);
			data += pitch;
			dst += width;
		}
	}
}
