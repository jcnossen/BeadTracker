/*

CPU only tracker

*/

#include "nivision.h"
#include "extcode.h"
#include "niimaq.h"

#include <Windows.h>
#include "utils.h"

#include "fftw-3.3.2/fftw3.h"

#define CALLCONV _FUNCC

typedef uchar pixel_t;

class CPUTracker
{
public:
	uint width, height;

	float *fft_in;
	fftwf_complex *fft_out;
	fftwf_plan fft_plan;

	CPUTracker(uint w,uint h) {
		width = w;
		height = h;

		fft_in = new float[w];
		fft_out = new fftwf_complex[h];

		fft_plan = fftwf_plan_dft_r2c_1d(w, fft_in, fft_out, FFTW_ESTIMATE);
	}

	~CPUTracker()
	{
		fftwf_destroy_plan(fft_plan);

		delete[] fft_in;
		delete[] fft_out;
	}
};

DLL_EXPORT CPUTracker* CALLCONV create_cpu_tracker(uint w, uint h)
{
	return new CPUTracker(w,h);
}

DLL_EXPORT void CALLCONV destroy_cpu_tracker(CPUTracker* tracker)
{
	delete tracker;
}


template<typename TPixel>
vector2f ComputeCOM(TPixel* data, uint w,uint h, uint pitch)
{
	uint sum=0;
	uint momentX=0;
	uint momentY=0;

	for (uint y=0;y<h;y++)
		for(uint x=0;x<w;x++)
		{
			pixel_t v = ((TPixel*)((uchar*)data + y*pitch)) [x];
			sum += v;
			momentX += x*v;
			momentY += y*v;
		}
	vector2f com;
	com.x = momentX / (float)sum;
	com.y = momentY / (float)sum;
	return com;
}


template<typename TPixel>
float* bgcorrect(TPixel* data, uint w, uint h, uint srcpitch, float* pMedian)
{
	float* dst = new float[w*h];
	TPixel* sortbuf = new TPixel[w*h];
	for (uint y=0;y<h;y++) {
		for (uint x=0;x<w;x++) {
			sortbuf[y*w+x] = ((TPixel*)((uchar*)data + y*srcpitch)) [x];
			dst[y*w+x] = sortbuf[y*w+x];
		}
	}
	std::sort(sortbuf, sortbuf+(w*h));
	float median = sortbuf[w*h/2];
	for (uint k=0;k<w*h;k++) {
		float v = (dst[k]-median);
		dst[k]=v*v;
	}
	delete[] sortbuf;
	if (pMedian) *pMedian = median;
	return dst;
}

template<typename TPixel>
void normalize(TPixel* d, uint w,uint h)
{
	TPixel maxv = d[0];
	TPixel minv = d[0];
	for (uint k=0;k<w*h;k++) {
		maxv = max(maxv, d[k]);
		minv = min(minv, d[k]);
	}
	for (uint k=0;k<w*h;k++)
		d[k]=(d[k]-minv)/(maxv-minv);
}


void saveImage(float* data, uint w, uint h, const char* filename)
{
	float maxv = data[0];
	float minv = data[0];
	for (int k=0;k<w*h;k++) {
		maxv = max(maxv, data[k]);
		minv = min(minv, data[k]);
	}
	ushort *norm = new ushort[w*h];
	for (int k=0;k<w*h;k++)
		norm[k] = ((1<<16)-1) * (data[k]-minv) / (maxv-minv);
	Image* dst = imaqCreateImage(IMAQ_IMAGE_U16, 0);
	imaqSetImageSize(dst, w, h);
	imaqArrayToImage(dst, norm, w, h);
	delete[] norm;

	ImageInfo info;
	imaqGetImageInfo(dst, &info);
	int success = imaqWriteFile(dst, filename, 0);
	if (!success) {
		char *errStr = imaqGetErrorText(imaqGetLastError());
		std::string msg = SPrintf("IMAQ WriteFile error: %s\n", errStr);
		imaqDispose(errStr);
		dbgout(msg);
	}
	imaqDispose(dst);
}

DLL_EXPORT void CALLCONV localize_image(Image* img, float* COM, float* xcor,  float* median, Image* dbgImg)
{
	ImageInfo info;
	imaqGetImageInfo(img, &info);
	float* bgcorrected = 0;

	if (info.imageType == IMAQ_IMAGE_U8)
		bgcorrected = bgcorrect( (uchar*)info.imageStart, info.xRes, info.yRes, info.pixelsPerLine, median);
	else if(info.imageType == IMAQ_IMAGE_U16)
		bgcorrected = bgcorrect( (ushort*)info.imageStart, info.xRes, info.yRes, info.pixelsPerLine*2, median);

	if (bgcorrected) {
		vector2f com = ComputeCOM(bgcorrected, info.xRes, info.yRes, info.xRes*sizeof(float));
		
		if (dbgImg) {
			normalize(bgcorrected, info.xRes, info.yRes);
			uchar* cv = new uchar[info.xRes*info.yRes];
			for (int k=0;k<info.xRes*info.yRes;k++)
				cv[k]= (uchar)(255.0f * bgcorrected[k]);
			imaqArrayToImage(dbgImg, cv, info.xRes, info.yRes);
			delete[] cv;
		}

		COM[0] = com.x;
		COM[1] = com.y;

		delete[] bgcorrected;
	}
}

