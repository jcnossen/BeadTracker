/*

CPU only tracker

*/

#include <Windows.h>

#include "cpu_tracker.h"
#include "../cudatrack/LsqQuadraticFit.h"
#include "random_distr.h"

const float XCorScale = 1.0f; // keep this at 1, because linear oversampling was obviously a bad idea..


static int round(xcor_t f) { return (int)(f+0.5f); }

CPUTracker::CPUTracker(int w, int h, int xcorwindow)
{
	width = w;
	height = h;
	xcorw = 0;
	fft_out = 0;
	fft_revout = 0;
	fft_plan_fw = fft_plan_bw = 0;
	
	srcImage = new float [w*h];
	debugImage = new float [w*h];
	memset(srcImage, 0, sizeof(float)*w*h);
	memset(debugImage, 0, sizeof(float)*w*h);

	zlut = 0;
	zlut_planes = zlut_res = 0;

	xcorProfileWidth = min(32, xcorwindow);

	setXCorWindow(xcorwindow);
}

void CPUTracker::setXCorWindow(int xcorwindow)
{
	if (xcorw!=xcorwindow) {
#ifdef TRK_USE_DOUBLE
		if (fft_plan_fw) fftw_destroy_plan(fft_plan_fw);
		if (fft_plan_bw) fftw_destroy_plan(fft_plan_bw);
#else
		if (fft_plan_fw) fftwf_destroy_plan(fft_plan_fw);
		if (fft_plan_bw) fftwf_destroy_plan(fft_plan_bw);
#endif

		delete[] fft_out;
		delete[] fft_revout;
	}
	xcorw = xcorwindow;

	if (xcorw>0) {
		X_xcr.resize(xcorw);
		Y_xcr.resize(xcorw);
		X_xc.resize(xcorw);
		X_result.resize(xcorw);
		Y_xc.resize(xcorw);
		Y_result.resize(xcorw);
		shiftedResult.resize(xcorw);

		fft_out = new complexc[xcorw];
		fft_revout = new complexc[xcorw];
#ifdef TRK_USE_DOUBLE
		fft_plan_fw = fftw_plan_dft_r2c_1d(xcorw, &X_xc[0], (fftw_complex*) fft_out, FFTW_ESTIMATE);
		fft_plan_bw = fftw_plan_dft_c2r_1d(xcorw, (fftw_complex*)fft_out, &X_result[0], FFTW_ESTIMATE);
#else
		fft_plan_fw = fftwf_plan_dft_r2c_1d(xcorw, &X_xc[0], (fftwf_complex*) fft_out, FFTW_ESTIMATE);
		fft_plan_bw = fftwf_plan_dft_c2r_1d(xcorw, (fftwf_complex*)fft_out, &X_result[0], FFTW_ESTIMATE);
#endif
	}
}

CPUTracker::~CPUTracker()
{
	setXCorWindow(0);
	delete[] srcImage;
	delete[] debugImage;
	if (zlut) delete[] zlut;
}

const inline float interp(float a, float b, float x) { return a + (b-a)*x; }

float CPUTracker::Interpolate(float x,float y)
{
	int rx=x, ry=y;
	float v00 = getPixel(rx,ry);
	float v10 = getPixel(rx+1,ry);
	float v01 = getPixel(rx,ry+1);
	float v11 = getPixel(rx+1,ry+1);

	float v0 = interp (v00, v10, x-rx);
	float v1 = interp (v01, v11, x-rx);

	return interp (v0, v1, y-ry);
}

#ifdef _DEBUG
	#define MARKPIXEL(x,y) (debugImage[ (int)(y)*width+ (int) (x)]+=maxValue*0.1f)
	#define MARKPIXELI(x,y) _markPixels(x,y,debugImage, width, maxValue*0.1f);
static void _markPixels(float x,float y, float* img, int w, float mv)
{
	img[ (int)floorf(y)*w+(int)floorf(x) ] += mv;
	img[ (int)floorf(y)*w+(int)ceilf(x) ] += mv;
	img[ (int)ceilf(y)*w+(int)floorf(x) ] += mv;
	img[ (int)ceilf(y)*w+(int)ceilf(x) ] += mv;
}
#else
	#define MARKPIXEL(x,y)
	#define MARKPIXELI(x,y)
#endif

vector2f CPUTracker::ComputeXCorInterpolated(vector2f initial, int iterations)
{
	// extract the image
	vector2f pos = initial;

#ifdef _DEBUG
	std::copy(srcImage, srcImage+width*height, debugImage);
	float maxValue = *std::max_element(srcImage,srcImage+width*height);
#endif

	for (int k=0;k<iterations;k++) {
		float xmin = pos.x - XCorScale * xcorw/2;
		float ymin = pos.y - XCorScale * xcorw/2;

		if (xmin < 0 || ymin < 0 || xmin+xcorw*XCorScale>=width || ymin+xcorw*XCorScale>=height) {
			vector2f z={};
			return z;
		}

		// generate X position xcor array (summing over y range)
		for (int x=0;x<xcorw;x++) {
			xcor_t s = 0.0f;
			for (int y=0;y<xcorProfileWidth;y++) {
				float xp = x * XCorScale + xmin;
				float yp = pos.y + XCorScale * (y - xcorProfileWidth/2);
				s += Interpolate(xp, yp);
				MARKPIXELI(xp, yp);
			}
			X_xc [x] = s;
			X_xcr [xcorw-x-1] = X_xc[x];
		}

		XCorFFTHelper(&X_xc[0], &X_xcr[0], &X_result[0]);
		xcor_t offsetX = ComputeMaxInterp(X_result) - (xcor_t)xcorw/2; //ComputeMaxInterp(X_result) - (float)xcorw/2 - 1;

		// generate Y position xcor array (summing over x range)
		for (int y=0;y<xcorw;y++) {
			xcor_t s = 0.0f; 
			for (int x=0;x<xcorProfileWidth;x++) {
				float xp = pos.x + XCorScale * (x - xcorProfileWidth/2);
				float yp = y * XCorScale + ymin;
				s += Interpolate(xp, yp);
				MARKPIXELI(xp,yp);
			}
			Y_xc[y] = s;
			Y_xcr [xcorw-y-1] = Y_xc[y];
		}

		XCorFFTHelper(&Y_xc[0], &Y_xcr[0], &Y_result[0]);
		xcor_t offsetY = ComputeMaxInterp(Y_result) - (xcor_t)xcorw/2;

		pos.x += (offsetX - 1) * XCorScale * 0.5f;
		pos.y += (offsetY - 1) * XCorScale * 0.5f;
	}

	return pos;
}





vector2f CPUTracker::ComputeXCor(vector2f initial)
{
	// extract the image
	vector2f pos = initial;

#ifdef _DEBUG
	std::copy(srcImage, srcImage+width*height, debugImage);
	float maxValue = *std::max_element(srcImage,srcImage+width*height);
#endif

	int rx = round(pos.x);
	int ry = round(pos.y);

	int xmin = rx - xcorw/2;
	int ymin = ry - xcorw/2;

	if (xmin < 0 || ymin < 0 || xmin+xcorw/2>=width || ymin+xcorw/2>=height) {
		vector2f z={};
		return z;
	}

	// generate X position xcor array (summing over y range)
	for (int x=0;x<xcorw;x++) {
		xcor_t s = 0.0f;
		for (int y=0;y<xcorProfileWidth;y++) {
			int xp = rx + x - xcorw/2;
			int yp = ry + y - xcorProfileWidth/2;
			s += getPixel(xp, yp);
			MARKPIXEL(xp, yp);
		}
		X_xc [x] = s;
		X_xcr [xcorw-x-1] = X_xc[x];
	}

	XCorFFTHelper(&X_xc[0], &X_xcr[0], &X_result[0]);
	xcor_t offsetX = ComputeMaxInterp(X_result) - (xcor_t)xcorw/2; 

	// generate Y position xcor array (summing over x range)
	for (int y=0;y<xcorw;y++) {
		xcor_t s = 0.0f; 
		for (int x=0;x<xcorProfileWidth;x++) {
			int xp = rx + x - xcorProfileWidth/2;
			int yp = ry + y - xcorw/2;
			s += getPixel(xp,yp);
			MARKPIXEL(xp,yp);
		}
		Y_xc[y] = s;
		Y_xcr [xcorw-y-1] = Y_xc[y];
	}

	XCorFFTHelper(&Y_xc[0], &Y_xcr[0], &Y_result[0]);
	xcor_t offsetY = ComputeMaxInterp(Y_result) - (xcor_t)xcorw/2;

	pos.x = rx + (offsetX - 1) * 0.5f;
	pos.y = ry + (offsetY - 1) * 0.5f;

	return pos;
}

void CPUTracker::OutputDebugInfo()
{
	for (int i=0;i<xcorw;i++) {
		//dbgout(SPrintf("i=%d,  X = %f;  X_rev = %f;  Y = %f,  Y_rev = %f\n", i, X_xc[i], X_xcr[i], Y_xc[i], Y_xcr[i]));
		dbgout(SPrintf("i=%d,  X_result = %f;   X = %f;  X_rev = %f\n", i, X_result[i], X_xc[i], X_xcr[i]));
	}
}



void CPUTracker::XCorFFTHelper(xcor_t* xc, xcor_t *xcr, xcor_t* result)
{
#ifdef TRK_USE_DOUBLE
	fftw_execute_dft_r2c(fft_plan_fw, xc, (fftw_complex*)fft_out);
	fftw_execute_dft_r2c(fft_plan_fw, xcr, (fftw_complex*)fft_revout);
#else
	fftwf_execute_dft_r2c(fft_plan_fw, xc, (fftwf_complex*)fft_out);
	fftwf_execute_dft_r2c(fft_plan_fw, xcr, (fftwf_complex*)fft_revout);
#endif

	// Multiply with conjugate of reverse
	for (int x=0;x<xcorw;x++) {
		fft_out[x] *= complexc(fft_revout[x].real(), -fft_revout[x].imag());
	}

#ifdef TRK_USE_DOUBLE
	fftw_execute_dft_c2r(fft_plan_bw, (fftw_complex*)fft_out, &shiftedResult[0]);
#else
	fftwf_execute_dft_c2r(fft_plan_bw, (fftwf_complex*)fft_out, &shiftedResult[0]);
#endif

	for (int x=0;x<xcorw;x++)
		result[x] = shiftedResult[ (x+xcorw/2) % xcorw ];
}



vector2f CPUTracker::ComputeCOM(float median)
{
	float sum=0;
	float momentX=0;
	float momentY=0;

	for (int y=0;y<height;y++)
		for(int x=0;x<width;x++)
		{
			float v = getPixel(x,y)-median;
			v *= v;
			sum += v;
			momentX += x*v;
			momentY += y*v;
		}
	vector2f com;
	com.x = momentX / (float)sum;
	com.y = momentY / (float)sum;
	return com;
}


float CPUTracker::ComputeMedian()
{
	return ::ComputeMedian(srcImage, width, height, width * sizeof(float), 0);
}


void CPUTracker::Normalize(float* d)
{
	if (!d) d=srcImage;
	normalize(d, width, height);
}


void CPUTracker::ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float range, vector2f center)
{
	if (radialDirs.size() != angularSteps) {
		radialDirs.resize(angularSteps);
		for (int j=0;j<angularSteps;j++) {
			float ang = 2*3.141593f*j/(float)angularSteps;
			vector2f d = { cosf(ang), sinf(ang) };
			radialDirs[j] = d;
		}
	}

	for (int i=0;i<radialSteps;i++)
		dst[i]=0.0f;

	float total = 0.0f;
	float rstep = range / radialSteps;
	for (int i=0;i<radialSteps; i++) {
		float sum = 0.0f;

		for (int a=0;a<angularSteps;a++) {
			float x = center.x + radialDirs[a].x * rstep*i;
			float y = center.y + radialDirs[a].y * rstep*i;
			sum += Interpolate(x,y)*i;
		}

		dst[i] = sum/(float)angularSteps;
		total += dst[i];
	}
	for (int i=0;i<radialSteps;i++)
		dst[i] /= total;
}

void CPUTracker::SetZLUT(float* data, int planes, int res)
{
	if (zlut) delete[] zlut;
	zlut = new float[planes*res];
	memcpy(zlut, data, sizeof(float)*planes*res);
	zlut_planes = planes;
	zlut_res = res;
}



float CPUTracker::ComputeZ(vector2f center, int angularSteps, float radius)
{
	// Compute the radial profile
	if (rprof.size() != zlut_res)
		rprof.resize(zlut_res);

	ComputeRadialProfile(&rprof[0], zlut_res, angularSteps, radius, center);

	// Now compare the radial profile to the profiles stored in Z
	if (rprof_diff.size() != zlut_planes)
		rprof_diff.resize(zlut_planes);
	for (int k=0;k<zlut_planes;k++) {
		float diffsum = 0.0f;
		for (int r = 0; r<zlut_res;r++) {
			float diff = rprof[r]-zlut[k*zlut_res+r];
			diffsum += diff*diff;
		}
		rprof_diff[k] = -diffsum;
	}

	return ComputeMaxInterp(rprof_diff);
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


void GenerateTestImage(CPUTracker* tracker, float xp, float yp, float size, float MaxPhotons)
{
	int w=tracker->GetWidth();
	int h=tracker->GetHeight();
	float S = 1.0f/size;
	float *d =  tracker->srcImage; //new float[tracker->width*tracker->height];
	for (int y=0;y<h;y++) {
		for (int x=0;x<w;x++) {
			float X = x - xp;
			float Y = y - yp;
			float r = sqrtf(X*X+Y*Y)+1;
			float v = sinf(r/(5*S)) * expf(-r*r*S*0.01f);
			d[y*w+x] = v;
		}
	}

	if (MaxPhotons>0) {
		tracker->Normalize();
		for (int k=0;k<w*h;k++) {
			float r = rand_poisson(d[k]*MaxPhotons);
			d[k] = r;
		}
	}
	tracker->Normalize();
}
