#pragma once

#include "std_incl.h" 

template<typename T> bool isNAN(const T& v) { 
	return !(v == v); 
}

void dbgout(const std::string& s);
std::string SPrintf(const char *fmt, ...);
CDLL_EXPORT void dbgprintf(const char *fmt,...);

template<typename T>
void DeleteAllElems(T& c) {
	for(typename T::iterator i=c.begin();i!=c.end();++i)
		delete *i;
	c.clear();
}


template<typename TPixel>
void normalize(TPixel* d, uint w,uint h)
{
	TPixel maxv = d[0];
	TPixel minv = d[0];
	for (uint k=0;k<w*h;k++) {
		maxv = std::max(maxv, d[k]);
		minv = std::min(minv, d[k]);
	}
	for (uint k=0;k<w*h;k++)
		d[k]=(d[k]-minv)/(maxv-minv);
}

inline float Lerp(float a, float b, float x) { return a + (b-a)*x; }

inline float Interpolate(float* image, int width, int height, float x,float y, bool* outside=0)
{
	int rx=x, ry=y;
	if (rx<0 || ry <0 || rx+1 >= width || ry+1>=height) {
		if (outside) *outside=true;
		return 0.0f;
	}
	if (outside) *outside=false;

	float v00 = image[width*ry+rx];
	float v10 = image[width*ry+rx+1];
	float v01 = image[width*(ry+1)+rx];
	float v11 = image[width*(ry+1)+rx+1];

	float v0 = Lerp(v00, v10, x-rx);
	float v1 = Lerp(v01, v11, x-rx);

	return Lerp(v0, v1, y-ry);
}


struct ImageData {
	float* data;
	int w,h;
	ImageData(float *d, int w, int h) : data(d), w(w),h(h) {}
	float& at(int x, int y) { return data[w*y+x]; }
	float interpolate(float x, float y, bool *outside=0) { return Interpolate(data, w,h, x,y,outside); }
	int numPixels() { return w*h; }
	void normalize() { ::normalize(data,w,h); }
	float mean() {
		float s=0.0f;
		for(int x=0;x<w*h;x++)
			s+=data[x];
		return s/(w*h);
	}
	float& operator[](int i) { return data[i]; }
};

float ComputeMostOccuringValue(ImageData& img); // a poor man's median

void GenerateTestImage(ImageData img, float xp, float yp, float size, float MaxPhotons);

// Corrected Radial Profile
float ComputeBgCorrectedCOM1D(float *data, int len, float cf=2.0f);
void ComputeCRP(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius, vector2f center, ImageData* src,float mean, float*crpmap=0);
void ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float minradius, float maxradius, vector2f center, ImageData* src, float mean);
void GenerateImageFromLUT(ImageData* image, ImageData* zlut, float minRadius, float maxRadius, vector2f pos, float z, float M);
void ApplyPoissonNoise(ImageData& img, float factor);
void ApplyGaussianNoise(ImageData& img, float sigma);
void WriteImageAsCSV(const char* file, float* d, int w,int h, const char *labels[]=0);
void WriteComplexImageAsCSV(const char* file, std::complex<float>* d, int w,int h, const char *labels[]=0);

int ReadJPEGFile(uchar* srcbuf, int srclen, uchar** data, int* width, int*height);
void WriteJPEGFile(uchar* data,int w,int h, const char * filename, int quality);

std::vector<uchar> ReadToByteBuffer(const char* filename);
double GetPreciseTime();

template<typename T>
void floatToNormalizedInt(T* dst, float *src, uint w,uint h, T maxValue)
{
	float maxv = src[0];
	float minv = src[0];
	for (uint k=0;k<w*h;k++) {
		maxv = std::max(maxv, src[k]);
		minv = std::min(minv, src[k]);
	}
	for (uint k=0;k<w*h;k++)
		dst[k] = maxValue * (src[k]-minv) / (maxv-minv);
}


template<typename T>
T* floatToNormalizedInt(float *src, uint w,uint h, T maxValue)
{ 
	T* r = new T[w*h]; 
	floatToNormalizedInt(r,src,w,h, maxValue);
	return r; 
}

template<typename T>
T ComputeStdDev(T* data, int len)
{
	T sum = 0.0, sum2=0.0;
	for (int a=0;a<len;a++) {
		sum+=data[a];
		sum2+=data[a]*data[a];
	}
	T mean = sum / len;
	return sqrt(sum2 / len- mean * mean);
}


/*
template<typename T, T minval, T maxval, int NBins>
T ComputeMostOccuringValue(T* data, int size)
{
	T maxv,minv;
	maxv=minv=img[0];
	for (int i=1;i<size;i++) {
		if (data[i] > maxv) maxv = data[i];
		if (data[i] < minv) minv = data[i];
	}
	const int NBins=256;
	int bins[NBins];
	for (int i=0;i<NBins;i++)
		bins[i]=0;
	for (int i=0;i<size;i++) {
		T v = data[i];
		int bin = NBins * (v - minv) / (maxv-minv);
		if (bin < 0) bin = 0;
		if (bin >= NBins) bin--;
		bins[bin]++;
	}
	int maxbinv= bins[0], maxbinIndex=0;
	for (int i=0;i<NBins;i++)
		if (bins[i]<maxbinv) {
			maxbinIndex = i;
			maxbinv = bins[i];
		}
	return 
}

*/
