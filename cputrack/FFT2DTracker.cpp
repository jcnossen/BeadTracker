
#include "Tracker.h" 
#include "FFT2DTracker.h" 

#include "jama_lu.h"


FFT2DTracker::FFT2DTracker(int w,int h)
{
	width = w;
	height = h;

	fft_buf = new complexc[w*h];
	fft_buf_mirrored = new complexc[w*h];
	mirror2D = new float[w*h];

	plan_fw2D = plan_bw2D = 0;
}

FFT2DTracker::~FFT2DTracker()
{
	if (plan_fw2D) {
		fftwf_destroy_plan(plan_fw2D);
		fftwf_destroy_plan(plan_bw2D);
	}
}

vector2f FFT2DTracker::ComputeXCor(float* image)
{
	vector2f pos;

	if (plan_fw2D == 0) {
		plan_fw2D = fftwf_plan_dft_r2c_2d(width, height, image, (fftwf_complex*)fft_buf, FFTW_MEASURE);
		plan_bw2D = fftwf_plan_dft_c2r_2d(width, height, (fftwf_complex*)fft_buf, mirror2D, FFTW_MEASURE);
	}

	// Mirror the image
	for (int y=0;y<height;y++) {
		for (int x=0;x<width;x++) 
			mirror2D[y*width+x] = image[(height-y-1)*width+width-x-1];
	}

	// Fourier transform
	fftwf_execute_dft_r2c(plan_fw2D, image, (fftwf_complex*)fft_buf);
	fftwf_execute_dft_r2c(plan_fw2D, mirror2D, (fftwf_complex*)fft_buf_mirrored);

	// Multiply with complex conjugate
	for (int k=0;k<width*height;k++) {
		complexc m = fft_buf_mirrored[k];
		fft_buf[k] *= complexc(m.real(), -m.imag());
	}

	// Transform back to spatial domain
	fftwf_execute_dft_c2r(plan_bw2D, (fftwf_complex*)fft_buf, mirror2D);

	// Compute interpolated maximum of mirror2D
	vector2f maxInterp = ComputeMax2DInterpolated(mirror2D);

	pos.x = width/2 - maxInterp.x * .5f;
	pos.y = height/2 - maxInterp.y * .5f;
	return pos;
}

vector2f FFT2DTracker::ComputeMax2DInterpolated(float* img)
{
	int maxX = 0, maxY = 0;
	float m = img[0];

	for (int y=0;y<height;y++){
		for (int x=0;x<width;x++) {
			float v = img[y*width+x];
			if (v > m) {
				m = v;
				maxX = x; maxY = y;
			}
		}
	}

	vector2f r;
	r.x = maxX;
	r.y = maxY;
	// todo: interpolation
	return r;
}


