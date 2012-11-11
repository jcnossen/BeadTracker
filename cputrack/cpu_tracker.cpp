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
static complexc conjugate(const complexc &v) { return complexc(v.real(),-v.imag()); }

CPUTracker::CPUTracker(int w, int h, int xcorwindow)
{
	width = w;
	height = h;

	xcorBuffer = 0;
	
	srcImage = new float [w*h];
	debugImage = new float [w*h];
	std::fill(srcImage, srcImage+w*h, 0.0f);
	std::fill(debugImage, debugImage+w*h, 0.0f);

	zluts = 0;
	zlut_planes = zlut_res = zlut_count = zlut_angularSteps = 0;
	zprofile_radius = 0.0f;
	xcorw = xcorwindow;

	qi_radialsteps = 0;
	qi_fft_forward = qi_fft_backward = 0;
}

CPUTracker::~CPUTracker()
{
	delete[] srcImage;
	delete[] debugImage;
	if (zluts && zlut_memoryOwner) 
		delete[] zluts;

	if (xcorBuffer)
		delete xcorBuffer;

	if (qi_fft_forward) { 
		delete qi_fft_forward;
		delete qi_fft_backward;
	}
}

void CPUTracker::SetImageFloat(float *src) {
	for (int k=0;k<width*height;k++)
		srcImage[k]=src[k];
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

XCor1DBuffer::XCor1DBuffer(int xcorw) 
	: fft_forward(xcorw, false), fft_backward(xcorw, true)
{
	X_xcr.resize(xcorw);
	Y_xcr.resize(xcorw);
	X_xc.resize(xcorw);
	X_result.resize(xcorw);
	Y_xc.resize(xcorw);
	Y_result.resize(xcorw);
	shiftedResult.resize(xcorw);
	this->xcorw = xcorw;

	fft_out = new complexc[xcorw];
	fft_revout = new complexc[xcorw];
}

XCor1DBuffer::~XCor1DBuffer()
{
	delete[] fft_out;
	delete[] fft_revout;
}



void XCor1DBuffer::OutputDebugInfo()
{
	for (int i=0;i<xcorw;i++) {
		//dbgout(SPrintf("i=%d,  X = %f;  X_rev = %f;  Y = %f,  Y_rev = %f\n", i, X_xc[i], X_xcr[i], Y_xc[i], Y_xcr[i]));
		dbgout(SPrintf("i=%d,  X_result = %f;   X = %f;  X_rev = %f\n", i, X_result[i], X_xc[i], X_xcr[i]));
	}
}



void XCor1DBuffer::XCorFFTHelper(complexc* xc, complexc *xcr, xcor_t* result)
{
	fft_forward.transform(xc, fft_out);
	fft_forward.transform(xcr, fft_revout);

	// Multiply with conjugate of reverse
	for (int x=0;x<xcorw;x++) {
		fft_out[x] *= conjugate(fft_revout[x]);
	}

	fft_backward.transform(fft_out, &shiftedResult[0]);

	for (int x=0;x<xcorw;x++)
		result[x] = shiftedResult[ (x+xcorw/2) % xcorw ].real();
}

vector2f CPUTracker::ComputeXCorInterpolated(vector2f initial, int iterations, int profileWidth)
{
	// extract the image
	vector2f pos = initial;

	if (!xcorBuffer)
		xcorBuffer = new XCor1DBuffer(xcorw);

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

		complexc* xc = &xcorBuffer->X_xc[0];
		complexc* xcr = &xcorBuffer->X_xcr[0];
		// generate X position xcor array (summing over y range)
		for (int x=0;x<xcorw;x++) {
			xcor_t s = 0.0f;
			for (int y=0;y<profileWidth;y++) {
				float xp = x * XCorScale + xmin;
				float yp = pos.y + XCorScale * (y - profileWidth/2);
				s += Interpolate(srcImage, width, height, xp, yp);
				MARKPIXELI(xp, yp);
			}
			xc [x] = s;
			xcr [xcorw-x-1] = xc[x];
		}

		xcorBuffer->XCorFFTHelper(xc, xcr, &xcorBuffer->X_result[0]);
		xcor_t offsetX = ComputeMaxInterp(&xcorBuffer->X_result[0],xcorBuffer->X_result.size()) - (xcor_t)xcorw/2;

		// generate Y position xcor array (summing over x range)
		xc = &xcorBuffer->Y_xc[0];
		xcr = &xcorBuffer->Y_xcr[0];
		for (int y=0;y<xcorw;y++) {
			xcor_t s = 0.0f; 
			for (int x=0;x<profileWidth;x++) {
				float xp = pos.x + XCorScale * (x - profileWidth/2);
				float yp = y * XCorScale + ymin;
				s += Interpolate(srcImage,width,height, xp, yp);
				MARKPIXELI(xp,yp);
			}
			xc[y] = s;
			xcr [xcorw-y-1] = xc[y];
		}

		xcorBuffer->XCorFFTHelper(xc,xcr, &xcorBuffer->Y_result[0]);
		xcor_t offsetY = ComputeMaxInterp(&xcorBuffer->Y_result[0], xcorBuffer->Y_result.size()) - (xcor_t)xcorw/2;

		pos.x += (offsetX - 1) * XCorScale * 0.5f;
		pos.y += (offsetY - 1) * XCorScale * 0.5f;
	}

	return pos;
}



vector2f CPUTracker::ComputeXCor(vector2f initial, int profileWidth)
{
	// extract the image
	vector2f pos = initial;

	if (!xcorBuffer)
		xcorBuffer = new XCor1DBuffer(xcorw);

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

	complexc* xc = &xcorBuffer->X_xc[0];
	complexc* xcr = &xcorBuffer->X_xcr[0];
	// generate X position xcor array (summing over y range)
	for (int x=0;x<xcorw;x++) {
		xcor_t s = 0.0f;
		for (int y=0;y<profileWidth;y++) {
			int xp = rx + x - xcorw/2;
			int yp = ry + y - profileWidth/2;
			s += getPixel(xp, yp);
			MARKPIXEL(xp, yp);
		}
		xc [x] = s;
		xcr [xcorw-x-1] = xc[x];
	}

	xcorBuffer->XCorFFTHelper(xc, xcr, &xcorBuffer->X_result[0]);
	xcor_t offsetX = ComputeMaxInterp(&xcorBuffer->X_result[0],xcorBuffer->X_result.size()) - (xcor_t)xcorw/2;

	// generate Y position xcor array (summing over x range)
	for (int y=0;y<xcorw;y++) {
		xcor_t s = 0.0f;
		for (int x=0;x<profileWidth;x++) {
			int xp = rx + x - profileWidth/2;
			int yp = ry + y - xcorw/2;
			s += getPixel(xp,yp);
			MARKPIXEL(xp,yp);
		}
		xc[y] = s;
		xcr [xcorw-y-1] = xc[y];
	}

	xcorBuffer->XCorFFTHelper(xc,xcr, &xcorBuffer->Y_result[0]);
	xcor_t offsetY = ComputeMaxInterp(&xcorBuffer->Y_result[0], xcorBuffer->Y_result.size()) - (xcor_t)xcorw/2;

	pos.x = rx + (offsetX - 1) * 0.5f;
	pos.y = ry + (offsetY - 1) * 0.5f;

	return pos;
}


vector2f CPUTracker::ComputeQI(vector2f initial, int iterations, int radialSteps, int angularStepsPerQ, float minRadius, float maxRadius)
{
	int nr=radialSteps;
	/*
	Compute profiles for each quadrant
	*/
	if (angularStepsPerQ != quadrantDirs.size()) {
		quadrantDirs.resize(angularStepsPerQ);
		for (int j=0;j<angularStepsPerQ;j++) {
			float ang = 0.5*3.141593f*j/(float)angularStepsPerQ;
			vector2f d = { cosf(ang), sinf(ang) };
			quadrantDirs[j] = d;
		}
	}
	if(!qi_fft_forward || qi_radialsteps != nr) {
		if(qi_fft_forward) {
			delete qi_fft_forward;
			delete qi_fft_backward;
		}
		qi_radialsteps = nr;
		qi_fft_forward = new kissfft<float>(nr*2,false);
		qi_fft_backward = new kissfft<float>(nr*2,true);
	}

	float* buf = (float*)ALLOCA(sizeof(float)*nr*4);
	float* q0=buf, *q1=buf+nr, *q2=buf+nr*2, *q3=buf+nr*3;
	complexc* concat0 = (complexc*)ALLOCA(sizeof(complexc)*nr*2);
	complexc* concat1 = concat0 + nr;

	vector2f center = initial;

	for (int k=0;k<iterations;k++){
		for (int q=0;q<4;q++) {
			ComputeQuadrantProfile(buf+q*nr, nr, angularStepsPerQ, q, minRadius, maxRadius, center);
		}
		
		// Build Ix = qL(-r) || qR(r)
		// qL = q1 + q2   (concat0)
		// qR = q0 + q3   (concat1)
		for(int r=0;r<nr;r++) {
			concat0[nr-r-1] = q1[r]+q2[r];
			concat1[r] = q0[r]+q3[r];
		}

		float offsetX = QI_ComputeOffset(concat0, nr);

		// Build Iy = qB(-r) || qT(r)
		// qT = q0 + q1
		// qB = q2 + q3
		for(int r=0;r<nr;r++) {
			concat0[r] = q0[r]+q1[r];
			concat1[nr-r-1] = q2[r]+q3[r];
		}
		float offsetY = QI_ComputeOffset(concat0, nr);

	//	dbgprintf("[%d] OffsetX: %f, OffsetY: %f\n", k, offsetX, offsetY);

		center.x += offsetX;
		center.y += offsetY;
	}

	return center;
}


// Profile is complexc[nr*2]
float CPUTracker::QI_ComputeOffset(complexc* profile, int nr)
{
	complexc* reverse = (complexc*)ALLOCA(sizeof(complexc)*nr*2);
	complexc* fft_out = (complexc*)ALLOCA(sizeof(complexc)*nr*2);
	complexc* fft_out2 = (complexc*)ALLOCA(sizeof(complexc)*nr*2);

	for(int x=0;x<nr*2;x++)
		reverse[x] = profile[nr*2-1-x];

	qi_fft_forward->transform(profile, fft_out);
	qi_fft_forward->transform(reverse, fft_out2); // fft_out2 contains fourier-domain version of reverse profile

	// multiply with conjugate
	for(int x=0;x<nr*2;x++)
		fft_out[x] *= conjugate(fft_out2[x]);

	qi_fft_backward->transform(fft_out, fft_out2);
	// fft_out2 now contains the autoconvolution
	// convert it to float
	float* autoconv = (float*)ALLOCA(sizeof(float)*nr*2);
	for(int x=0;x<nr*2;x++)
		autoconv[x] = fft_out2[(x+nr)%(nr*2)].real();
	float maxPos = ComputeMaxInterp(autoconv, nr*2);
	float dr = (maxPos - nr) * 0.5f;
	return dr / (3.141593f * 0.5f);
}


void CPUTracker::ComputeQuadrantProfile(float* dst, int radialSteps, int angularSteps, int quadrant, float minRadius, float maxRadius, vector2f center)
{
	const int qmat[] = {
		1, 1,
		-1, 1,
		-1, -1,
		1, -1 };
	int mx = qmat[2*quadrant+0];
	int my = qmat[2*quadrant+1];

	for (int i=0;i<radialSteps;i++)
		dst[i]=0.0f;

	float total = 0.0f;
	float rstep = (maxRadius - minRadius) / radialSteps;
	for (int i=0;i<radialSteps; i++) {
		float sum = 0.0f;
		float r = minRadius + rstep * i;

		for (int a=0;a<angularSteps;a++) {
			float x = center.x + mx*quadrantDirs[a].x * r;
			float y = center.y + my*quadrantDirs[a].y * r;
			sum += Interpolate(srcImage,width,height, x,y);
		}

		dst[i] = sum;
		total += dst[i];
	}
	for (int i=0;i<radialSteps;i++)
		dst[i] /= total;
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


void CPUTracker::ComputeRadialProfile(float* dst, int radialSteps, int angularSteps, float radius, vector2f center)
{
	::ComputeRadialProfile(dst, radialSteps, angularSteps, radius, center, srcImage, width, height);
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
	if (xcorBuffer) {
		CopyCpxVector(xprof, xcorBuffer->X_xc);
		CopyCpxVector(yprof, xcorBuffer->Y_xc);
		xconv = xcorBuffer->X_result;
		yconv = xcorBuffer->Y_result;
	}
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



