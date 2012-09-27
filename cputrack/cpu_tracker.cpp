/*

CPU only tracker

*/

#include "nivision.h"
#include "extcode.h"
#include "niimaq.h"

#include <Windows.h>
#include "utils.h"

#include "fftw-3.3.2/fftw3.h"
#include <complex>

#define CALLCONV _FUNCC

/* lv_prolog.h and lv_epilog.h set up the correct alignment for LabVIEW data. */
#include "lv_prolog.h"
struct LVFloatArray {
	int32_t dimSize;
	float elt[1];
};
typedef LVFloatArray **TD1Hdl;
#include "lv_epilog.h"


typedef uchar pixel_t;
typedef std::complex<float> complexf;

class CPUTracker
{
public:
	uint width, height;

	complexf *fft_out, *fft_revout;
	float *result;
	fftwf_plan fft_plan_fw, fft_plan_bw;

	float* graphResult;

	uint xcorw;
	float *xc, *xc_r;

	CPUTracker(uint w,uint h);
	~CPUTracker();
	vector2f ComputeXCor(ushort* data, uint w, uint h, uint pitch, vector2f initial);
	void XCorFFTHelper();
	// Compute the interpolated index of the maximum value in the result array
	float ComputeMaxInterp();
};


CPUTracker::CPUTracker(uint w,uint h)
{
	width = w;
	height = h;

	xcorw = 64;
	xc = new float[xcorw];
	xc_r = new float[xcorw];

	graphResult = new float[xcorw];

	fft_revout = new complexf[xcorw];
	fft_out = new complexf[xcorw];
	result = new float[xcorw];

	fft_plan_fw = fftwf_plan_dft_r2c_1d(xcorw, xc, (fftwf_complex*) fft_out, FFTW_ESTIMATE);
	fft_plan_bw = fftwf_plan_dft_c2r_1d(xcorw, (fftwf_complex*)fft_out, result, FFTW_ESTIMATE);
}

CPUTracker::~CPUTracker()
{
	fftwf_destroy_plan(fft_plan_fw);
	fftwf_destroy_plan(fft_plan_bw);

	delete[] fft_revout;
	delete[] fft_out;
	delete[] result;

	delete[] xc;
	delete[] xc_r;

	delete[] graphResult;
}

vector2f CPUTracker::ComputeXCor(ushort* data, uint w,uint h,uint pitch, vector2f initial)
{
	// extract the image
	vector2f pos;

	int sl = 32;
	int xmid = (int)( initial.x );
	int ymid = (int)( initial.y );

	int xmin = initial.x - xcorw/2;
	int ymin = initial.y - xcorw/2;

	// generate X position xcor array (summing over y range)
	for (int x=0;x<xcorw;x++) {
		xc [x] = 0.0f;
		for (int y=ymid-sl;y<ymid+sl;y++)
			xc [x] += ((ushort*) ((uchar*)data + (y*pitch)))[x + xmin];
		xc_r [xcorw-x-1] = xc [x];

		graphResult[x] = xc[x] / ( sl*2 );
	}

	XCorFFTHelper();
	pos.x = initial.x + 0.5f * ComputeMaxInterp();

	// generate Y position xcor array (summing over x range)
	for (int y=0;y<xcorw;y++) {
		xc [y] = 0.0f;
		for (int x=xmid-sl;x<xmid+sl;x++) 
			xc [y] += ((ushort*) ((uchar*)data + ( (y+ymin)*pitch)))[x];
		xc_r [xcorw-y-1] = xc[y];
	}

	XCorFFTHelper();
	pos.y = initial.y + 0.5f * ComputeMaxInterp();
	return pos;
}

void CPUTracker::XCorFFTHelper()
{
	// need to optimize this: the DFT of the reverse sequence should be calculatable from the known DFT (right?)
	fftwf_execute_dft_r2c(fft_plan_fw, xc, (fftwf_complex*)fft_out);
	fftwf_execute_dft_r2c(fft_plan_fw, xc_r, (fftwf_complex*)fft_revout);

	// Multiply with conjugate of reverse
	for (int x=0;x<xcorw;x++) {
		fft_out[x] *= complexf(fft_revout[x].real(), -fft_revout[x].imag());
	}

	fftwf_execute_dft_c2r(fft_plan_bw, (fftwf_complex*)fft_out, result);
}

float CPUTracker::ComputeMaxInterp()
{
	uint iMax=0;
	float vMax=0;
	for (uint k=0;k<xcorw;k++) {
		if (result[k]>vMax) {
			vMax = result[k];
			iMax = k;
		}
	}

	return iMax;
}


template<typename TPixel>
vector2f ComputeCOM(TPixel* data, uint w,uint h)
{
	float sum=0;
	float momentX=0;
	float momentY=0;

	for (uint y=0;y<h;y++)
		for(uint x=0;x<w;x++)
		{
			pixel_t v = data[w*y+x];
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
		float v = dst[k]-median;
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

ushort* floatToNormalizedUShort(float *data, uint w,uint h)
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
	return norm;
}

void saveImage(float* data, uint w, uint h, const char* filename)
{
	ushort* d = floatToNormalizedUShort(data,w,h);
	Image* dst = imaqCreateImage(IMAQ_IMAGE_U16, 0);
	imaqSetImageSize(dst, w, h);
	imaqArrayToImage(dst, d, w, h);
	delete[] d;

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

DLL_EXPORT void CALLCONV generate_test_image(Image *img, uint w, uint h, float xp, float yp, float size)
{
	float S = 1.0f/size;
	float *d = new float[w*h];
	for (uint y=0;y<h;y++)
		for (uint x=0;x<w;x++) {
			float X = x - xp;
			float Y = y - yp;
			float r = sqrtf(X*X+Y*Y)+1;
			float v = 0.1 + sinf( (r-10)*2*3.141593f*S) / r;
			d[y*w+x] = v;
		}

	ushort* result = floatToNormalizedUShort(d, w, h);
	imaqArrayToImage(img, result, w,h);
	delete[] result;
}


DLL_EXPORT CPUTracker* CALLCONV create_cpu_tracker(uint w, uint h)
{
	return new CPUTracker(w,h);
}

DLL_EXPORT void CALLCONV destroy_cpu_tracker(CPUTracker* tracker)
{
	delete tracker;
}


DLL_EXPORT void CALLCONV copy_crosscorrelation_result(CPUTracker* tracker, TD1Hdl resultArray)
{
	if (!resultArray)
		return;

	LVFloatArray* dst = *resultArray;

	int len = min ( (*resultArray)->dimSize, tracker->xcorw );
	for (int i=0;i<tracker->xcorw;i++)
		dst->elt[i] = tracker->graphResult[i];
}

DLL_EXPORT void CALLCONV localize_image(CPUTracker* tracker, Image* img, float* COM, float* xcor,  float* median, Image* dbgImg)
{
	ImageInfo info;
	imaqGetImageInfo(img, &info);
	float* bgcorrected = 0;

	if (info.imageType == IMAQ_IMAGE_U8)
		bgcorrected = bgcorrect( (uchar*)info.imageStart, info.xRes, info.yRes, info.pixelsPerLine, median);
	else if(info.imageType == IMAQ_IMAGE_U16)
		bgcorrected = bgcorrect( (ushort*)info.imageStart, info.xRes, info.yRes, info.pixelsPerLine*2, median);

	if (bgcorrected) {
		normalize(bgcorrected, info.xRes, info.yRes);
		vector2f com = ComputeCOM(bgcorrected, info.xRes, info.yRes);
		
		if (dbgImg) {
			uchar* cv = new uchar[info.xRes*info.yRes];
			for (int k=0;k<info.xRes*info.yRes;k++)
				cv[k]= (uchar)(255.0f * bgcorrected[k]);
			imaqArrayToImage(dbgImg, cv, info.xRes, info.yRes);
			delete[] cv;
		}

		COM[0] = com.x;
		COM[1] = com.y;

		vector2f xcorpos = tracker->ComputeXCor( (ushort*)info.imageStart, info.xRes, info.yRes, info.pixelsPerLine*2, com);
		delete[] bgcorrected;

		xcor[0] = xcorpos.x;
		xcor[1] = xcorpos.y;
	}
}

