/*

CPU only tracker

*/

#include "nivision.h"
#include "extcode.h"
#include "niimaq.h"

#include <Windows.h>

#include "cpu_tracker.h"
#include "LsqQuadraticFit.h"

#define CALLCONV _FUNCC

/* lv_prolog.h and lv_epilog.h set up the correct alignment for LabVIEW data. */
#include "lv_prolog.h"
struct LVFloatArray {
	int32_t dimSize;
	float elt[1];
};
typedef LVFloatArray **TD1Hdl;
#include "lv_epilog.h"


const float XCorScale = 0.5f;

CPUTracker::CPUTracker(uint w,uint h)
{
	width = w;
	height = h;

	xcorw = 128;

	X_xc.resize(xcorw);
	X_xcr.resize(xcorw);
	X_result.resize(xcorw);
	Y_xc.resize(xcorw);
	Y_xcr.resize(xcorw);
	Y_result.resize(xcorw);

	fft_revout = new complexf[xcorw];
	fft_out = new complexf[xcorw];

	fft_plan_fw = fftwf_plan_dft_r2c_1d(xcorw, &X_xc[0], (fftwf_complex*) fft_out, FFTW_ESTIMATE);
	fft_plan_bw = fftwf_plan_dft_c2r_1d(xcorw, (fftwf_complex*)fft_out, &X_result[0], FFTW_ESTIMATE);

	srcImage = new float [w*h];
}

CPUTracker::~CPUTracker()
{
	fftwf_destroy_plan(fft_plan_fw);
	fftwf_destroy_plan(fft_plan_bw);

	delete[] fft_revout;
	delete[] fft_out;
	delete[] srcImage;
}

const inline float interp(float a, float b, float x) { return a + (b-a)*x; }

float CPUTracker::interpolate(float x,float y)
{
	int rx=x, ry=y;
	float v00 = getPixel(rx,ry);
	float v01 = getPixel(rx+1,ry);
	float v10 = getPixel(rx,ry+1);
	float v11 = getPixel(rx+1,ry+1);

	float v0 = interp (v00, v10, x-rx);
	float v1 = interp (v01, v11, x-rx);

	return interp (v0, v1, y-ry);
}

vector2f CPUTracker::ComputeXCor(vector2f initial)
{
	// extract the image
	float scale = (1.0f/(XCorProfileLen*float(1<<16)-1));

	float xmin = initial.x - XCorScale * xcorw/2;
	float ymin = initial.y - XCorScale * xcorw/2;

	// generate X position xcor array (summing over y range)
	for (uint x=0;x<xcorw;x++) {
		float s = 0.0f;
		for (int y=0;y<XCorProfileLen;y++)
			s += interpolate(x * XCorScale + xmin, y * XCorScale + initial.y - XCorScale * XCorProfileLen/2);
		X_xc [x] = s*scale;
		X_xcr [xcorw-x-1] = X_xc[x];
	}

	XCorFFTHelper(&X_xc[0], &X_xcr[0], &X_result[0]);
	float offsetX = ComputeMaxInterp(X_result) - (float)xcorw/2;

	// generate Y position xcor array (summing over x range)
	for (uint y=0;y<xcorw;y++) {
		float s = 0.0f; 
		for (int x=0;x<XCorProfileLen;x++) 
			s += interpolate(x * XCorScale + initial.x - XCorProfileLen/2 * XCorScale, y * XCorScale + ymin);
		Y_xc[y] = s*scale;
		Y_xcr [xcorw-y-1] = Y_xc[y];
	}

	XCorFFTHelper(&Y_xc[0], &Y_xcr[0], &Y_result[0]);
	float offsetY = ComputeMaxInterp(Y_result) - (float)xcorw/2;

	dbgout(SPrintf("offsetX: %f, offsetY: %f\n", offsetX, offsetY));

	vector2f pos;
	pos.x = initial.x + offsetX * XCorScale;
	pos.y = initial.y + offsetY * XCorScale;
	return pos;
}

void CPUTracker::XCorFFTHelper(float* xc, float* xcr, float* result)
{
	// need to optimize this: the DFT of the reverse sequence should be calculatable from the known DFT (right?)
	fftwf_execute_dft_r2c(fft_plan_fw, xc, (fftwf_complex*)fft_out);
	fftwf_execute_dft_r2c(fft_plan_fw, xcr, (fftwf_complex*)fft_revout);

	// Multiply with conjugate of reverse
	for (uint x=0;x<xcorw;x++) {
		fft_out[x] *= complexf(fft_revout[x].real(), -fft_revout[x].imag());
	}

	fftwf_execute_dft_c2r(fft_plan_bw, (fftwf_complex*)fft_out, xc);
	for (uint x=0;x<xcorw;x++)
		result[x] = xc[ (x+xcorw/2) % xcorw ];
}

float CPUTracker::ComputeMaxInterp(const std::vector<float>& r)
{
	uint iMax=0;
	float vMax=0;
	for (uint k=0;k<r.size();k++) {
		if (r[k]>vMax) {
			vMax = r[k];
			iMax = k;
		}
	}
	
	float xs[] = {-2, -1, 0, 1, 2};
	LsqSqQuadFit<float> qfit(5, xs, &r[iMax-2]);
	float interpMax = qfit.maxPos();

	return (float)iMax - interpMax * 0.5f;
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
			TPixel v = data[w*y+x];
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
void CPUTracker::bgcorrect(TPixel* data, uint w, uint h, uint srcpitch, float* pMedian)
{
	TPixel* sortbuf = new TPixel[w*h];
	for (uint y=0;y<h;y++) {
		for (uint x=0;x<w;x++) {
			sortbuf[y*w+x] = ((TPixel*)((uchar*)data + y*srcpitch)) [x];
			srcImage[y*w+x] = sortbuf[y*w+x];
		}
	}
	std::sort(sortbuf, sortbuf+(w*h));
	float median = sortbuf[w*h/2];
	for (uint k=0;k<w*h;k++) {
		float v = srcImage[k]-median;
		srcImage[k]=v*v;
	}
	delete[] sortbuf;
	if (pMedian) *pMedian = median;
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
	for (uint k=0;k<w*h;k++) {
		maxv = max(maxv, data[k]);
		minv = min(minv, data[k]);
	}
	ushort *norm = new ushort[w*h];
	for (uint k=0;k<w*h;k++)
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
			float v = 0.1 + sinf( (r-10)/5) * expf(-r*S);
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

void copyToLVArray (TD1Hdl r, const std::vector<float>& a)
{
	LVFloatArray* dst = *r;

	uint len = min( dst->dimSize, a.size () );
	for (uint i=0;i<a.size();i++)
		dst->elt[i] = a[i];
}

DLL_EXPORT void CALLCONV copy_crosscorrelation_result(CPUTracker* tracker, TD1Hdl x_result, TD1Hdl y_result, TD1Hdl x_xc, TD1Hdl y_xc)
{
	if (x_result) copyToLVArray (x_result, tracker->X_result);
	if (y_result) copyToLVArray (y_result, tracker->Y_result);
	if (x_xc) copyToLVArray (x_xc, tracker->X_xc);
	if (y_xc) copyToLVArray (y_xc, tracker->Y_xc);
}

DLL_EXPORT void CALLCONV localize_image(CPUTracker* tracker, Image* img, float* COM, float* xcor,  float* median, Image* dbgImg)
{
	ImageInfo info;
	imaqGetImageInfo(img, &info);

	if (info.xRes != tracker->width || info.yRes != tracker->height)
		return;

	if (info.imageType == IMAQ_IMAGE_U8)
		tracker->bgcorrect((uchar*)info.imageStart, info.xRes, info.yRes, info.pixelsPerLine, median);
	else if(info.imageType == IMAQ_IMAGE_U16)
		tracker->bgcorrect((ushort*)info.imageStart, info.xRes, info.yRes, info.pixelsPerLine*2, median);
	else 
		return;

	normalize(tracker->srcImage, info.xRes, info.yRes);
	vector2f com = ComputeCOM(tracker->srcImage, info.xRes, info.yRes);

	if (dbgImg) {
		uchar* cv = new uchar[info.xRes*info.yRes];
		for (int k=0;k<info.xRes*info.yRes;k++)
			cv[k]= (uchar)(255.0f * tracker->srcImage[k]);
		imaqArrayToImage(dbgImg, cv, info.xRes, info.yRes);
		delete[] cv;
	}

	COM[0] = com.x;
	COM[1] = com.y;

	vector2f xcorpos = tracker->ComputeXCor(com);
	xcor[0] = xcorpos.x;
	xcor[1] = xcorpos.y;
}

