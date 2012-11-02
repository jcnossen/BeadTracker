/*

CPU only tracker

*/

#pragma warning(disable: 4996) // Function call with parameters that may be unsafe

#include "cpu_tracker.h"
#include "LsqQuadraticFit.h"
#include "random_distr.h"
#include <exception>

const float XCorScale = 1.0f; // keep this at 1, because linear oversampling was obviously a bad idea..

static int round(xcor_t f) { return (int)(f+0.5f); }

CPUTracker::CPUTracker(int w, int h, int xcorwindow) : fft_forward(xcorwindow, false), fft_backward(xcorwindow, true)
{
	width = w;
	height = h;
	fft_out = 0;
	fft_revout = 0;
	
	srcImage = new float [w*h];
	debugImage = new float [w*h];
	std::fill(srcImage, srcImage+w*h, 0.0f);
	std::fill(debugImage, debugImage+w*h, 0.0f);

	zluts = 0;
	zlut_planes = zlut_res = zlut_count = zlut_angularSteps = 0;
	zprofile_radius = 0.0f;
	xcorw = xcorwindow;

	X_xcr.resize(xcorw);
	Y_xcr.resize(xcorw);
	X_xc.resize(xcorw);
	X_result.resize(xcorw);
	Y_xc.resize(xcorw);
	Y_result.resize(xcorw);
	shiftedResult.resize(xcorw);

	fft_out = new complexc[xcorw];
	fft_revout = new complexc[xcorw];
}

CPUTracker::~CPUTracker()
{
	xcorw=0;
	delete[] fft_out;
	delete[] fft_revout;
	delete[] srcImage;
	delete[] debugImage;
	if (zluts && zlut_memoryOwner) 
		delete[] zluts;
}

void CPUTracker::SetImageFloat(float *src) {
	for (int k=0;k<width*height;k++)
		srcImage[k]=src[k];
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

vector2f CPUTracker::ComputeXCorInterpolated(vector2f initial, int iterations, int profileWidth)
{
	// extract the image
	vector2f pos = initial;

	if (xcorw < profileWidth)
		profileWidth = xcorw;

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
			for (int y=0;y<profileWidth;y++) {
				float xp = x * XCorScale + xmin;
				float yp = pos.y + XCorScale * (y - profileWidth/2);
				s += Interpolate(xp, yp);
				MARKPIXELI(xp, yp);
			}
			X_xc [x] = s;
			X_xcr [xcorw-x-1] = X_xc[x];
		}

		XCorFFTHelper(&X_xc[0], &X_xcr[0], &X_result[0]);
		xcor_t offsetX = ComputeMaxInterp(&X_result[0],X_result.size()) - (xcor_t)xcorw/2;

		// generate Y position xcor array (summing over x range)
		for (int y=0;y<xcorw;y++) {
			xcor_t s = 0.0f; 
			for (int x=0;x<profileWidth;x++) {
				float xp = pos.x + XCorScale * (x - profileWidth/2);
				float yp = y * XCorScale + ymin;
				s += Interpolate(xp, yp);
				MARKPIXELI(xp,yp);
			}
			Y_xc[y] = s;
			Y_xcr [xcorw-y-1] = Y_xc[y];
		}

		XCorFFTHelper(&Y_xc[0], &Y_xcr[0], &Y_result[0]);
		xcor_t offsetY = ComputeMaxInterp(&Y_result[0], Y_result.size()) - (xcor_t)xcorw/2;

		pos.x += (offsetX - 1) * XCorScale * 0.5f;
		pos.y += (offsetY - 1) * XCorScale * 0.5f;
	}

	return pos;
}



vector2f CPUTracker::ComputeXCor(vector2f initial, int profileWidth)
{
	// extract the image
	vector2f pos = initial;

	if (xcorw < profileWidth)
		profileWidth = xcorw;

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
		for (int y=0;y<profileWidth;y++) {
			int xp = rx + x - xcorw/2;
			int yp = ry + y - profileWidth/2;
			s += getPixel(xp, yp);
			MARKPIXEL(xp, yp);
		}
		X_xc [x] = s;
		X_xcr [xcorw-x-1] = X_xc[x];
	}

	XCorFFTHelper(&X_xc[0], &X_xcr[0], &X_result[0]);
	xcor_t offsetX = ComputeMaxInterp(&X_result[0], X_result.size()) - (xcor_t)xcorw/2; 

	// generate Y position xcor array (summing over x range)
	for (int y=0;y<xcorw;y++) {
		xcor_t s = 0.0f; 
		for (int x=0;x<profileWidth;x++) {
			int xp = rx + x - profileWidth/2;
			int yp = ry + y - xcorw/2;
			s += getPixel(xp,yp);
			MARKPIXEL(xp,yp);
		}
		Y_xc[y] = s;
		Y_xcr [xcorw-y-1] = Y_xc[y];
	}

	XCorFFTHelper(&Y_xc[0], &Y_xcr[0], &Y_result[0]);
	xcor_t offsetY = ComputeMaxInterp(&Y_result[0], Y_result.size()) - (xcor_t)xcorw/2;

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



void CPUTracker::XCorFFTHelper(complexc* xc, complexc *xcr, xcor_t* result)
{
	fft_forward.transform(xc, fft_out);
	fft_forward.transform(xcr, fft_revout);

	// Multiply with conjugate of reverse
	for (int x=0;x<xcorw;x++) {
		fft_out[x] *= complexc(fft_revout[x].real(), -fft_revout[x].imag());
	}

	fft_backward.transform(fft_out, &shiftedResult[0]);

	for (int x=0;x<xcorw;x++)
		result[x] = shiftedResult[ (x+xcorw/2) % xcorw ].real();
}


vector2f CPUTracker::ComputeQI(vector2f initial, int iterations, int radialSteps, int angularStepsPerQ, float radius)
{
	/*
	Compute profiles for each quadrant

	*/

	if (angularStepsPerQ != quadrantDirs.size()) {
		for (int j=0;j<angularStepsPerQ;j++) {
			float ang = 2*3.141593f*j/(float)angularStepsPerQ;
			vector2f d = { cosf(ang), sinf(ang) };
			radialDirs[j] = d;
		}
	}


	return initial;
}


vector2f CPUTracker::ComputeBgCorrectedCOM()
{
	float sum=0, sum2=0;
	float momentX=0;
	float momentY=0;

	for (int y=0;y<height;y++)
		for (int x=0;x<width;x++) {
			float v = getPixel(x,y);
			sum += v;
			sum2 += v*v;
		}

	float invN = 1.0f/(width*height);
	float mean = sum * invN;
	float stdev = sqrtf(sum2 * invN - mean * mean);
	sum = 0.0f;

	for (int y=0;y<height;y++)
		for(int x=0;x<width;x++)
		{
			float v = getPixel(x,y);
			v = std::max(0.0f, fabs(v-mean)-stdev);
			sum += v;
			momentX += x*v;
			momentY += y*v;
		}
	vector2f com;
	com.x = momentX / (float)sum;
	com.y = momentY / (float)sum;
	return com;
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
			sum += Interpolate(x,y);
		}

		dst[i] = sum;
		total += dst[i];
	}
	for (int i=0;i<radialSteps;i++)
		dst[i] /= total;
}

void CPUTracker::SetZLUT(float* data, int planes, int res, int numLUTs, float prof_radius, int angularSteps, bool copyMemory)
{
	if (zluts && zlut_memoryOwner)
		delete[] zluts;

	if (copyMemory) {
		zluts = new float[planes*res*numLUTs];
		std::copy(data, data+(planes*res*numLUTs), zluts);
	} else
		zluts = data;
	zlut_memoryOwner = !copyMemory;
	zlut_planes = planes;
	zlut_res = res;
	zlut_count = numLUTs;
	zprofile_radius = prof_radius;
	zlut_angularSteps = angularSteps;
}



float CPUTracker::ComputeZ(vector2f center, int angularSteps, int zlutIndex)
{
	if (!zluts)
		return 0.0f;

	// Compute the radial profile
	if (rprof.size() != zlut_res)
		rprof.resize(zlut_res);

	ComputeRadialProfile(&rprof[0], zlut_res, angularSteps, zprofile_radius, center);

	// Now compare the radial profile to the profiles stored in Z
	if (rprof_diff.size() != zlut_planes)
		rprof_diff.resize(zlut_planes);

	float* zlut_sel = &zluts[zlut_planes*zlut_res*zlutIndex];

	for (int k=0;k<zlut_planes;k++) {
		float diffsum = 0.0f;
		for (int r = 0; r<zlut_res;r++) {
			float diff = rprof[r]-zlut_sel[k*zlut_res+r];
			diffsum += diff*diff;
		}
		rprof_diff[k] = -diffsum;
	}

	float z = ComputeMaxInterp(&rprof_diff[0], rprof_diff.size());
	return z / (float)(zlut_planes-1);
}

static void CopyCpxVector(std::vector<xcor_t>& xprof, const std::vector<complexc>& src) {
	xprof.resize(src.size());
	for (int k=0;k<src.size();k++)
		xprof[k]=src[k].imag();
}

bool CPUTracker::GetLastXCorProfiles(std::vector<xcor_t>& xprof, std::vector<xcor_t>& yprof, 
		std::vector<xcor_t>& xconv, std::vector<xcor_t>& yconv)
{
	CopyCpxVector(xprof, X_xc);
	CopyCpxVector(yprof, Y_xc);
	xconv = X_result;
	yconv = Y_result;
	return true;
}

void CPUTrackerImageBuffer::Assign(ushort* srcData, int pitch)
{
	uchar *d = (uchar*)srcData;
	for (int y=0;y<h;y++) {
		memcpy(&data[y*w], d, sizeof(ushort)*w);
		d += pitch;
	}
}




CPUTrackerImageBuffer::~CPUTrackerImageBuffer ()
{
	delete[] data;
}

vector2f CPUTracker::ComputeXCor2D()
{
	vector2f x={};
	return x;
}



