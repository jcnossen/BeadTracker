#include "../cputrack/cpu_tracker.h"
#include <Windows.h>
#include <stdint.h>
#include "../cputrack/random_distr.h"

double getPreciseTime()
{
	uint64_t freq, time;

	QueryPerformanceCounter((LARGE_INTEGER*)&time);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	return (double)time / (double)freq;
}

void GenerateTestImage(CPUTracker* tracker, float xp, float yp, float size, float MaxPhotons)
{
	int w=tracker->width;
	int h=tracker->height;
	float S = 1.0f/size;
	float *d =  tracker->srcImage; //new float[tracker->width*tracker->height];
	for (int y=0;y<h;y++) {
		for (int x=0;x<w;x++) {
			float X = x - xp;
			float Y = y - yp;
			float r = sqrtf(X*X+Y*Y)+1;
			float v = 0.1 + sinf( (r-10)/5) * expf(-r*S);
			d[y*w+x] = v;
		}
	}

	if (MaxPhotons>0) {
		tracker->Normalize();
		for (int k=0;k<w*h;k++) {
			d[k] = rand_poisson(d[k]*MaxPhotons);
		}
	}
	tracker->Normalize();
}

void Localize(CPUTracker* t, vector2f& com, vector2f& xcor)
{
	float median = ComputeMedian(t->srcImage, t->width, t->height, t->width*sizeof(float),0);
	com = t->ComputeCOM(median);
	vector2f initial = {com.x, com.y};
	xcor = t->ComputeXCor(initial);
}


void SpeedTest()
{
	int N = 500;
	CPUTracker tracker(150,150, 128);

	// Speed test
	vector2f comdist={}, xcordist={};
	double tloc = 0.0, tgen=0.0;
	for (int k=0;k<N;k++)
	{
		double t0= getPreciseTime();
		float xp = tracker.width/2+(rand_uniform<float>() - 0.5) * 5;
		float yp = tracker.height/2+(rand_uniform<float>() - 0.5) * 5;
		float size = 20.0f;

		GenerateTestImage(&tracker, xp, yp, size, 200);
		
		double t1 = getPreciseTime();
		vector2f com, xcor;
		Localize(&tracker, com, xcor);

		comdist.x += fabsf(com.x - xp);
		comdist.y += fabsf(com.y - yp);

		xcordist.x += fabsf(xcor.x - xp);
		xcordist.y += fabsf(xcor.y - yp);
		double t2 = getPreciseTime();

		tloc+=t2-t1;
		tgen+=t1-t0;
	}

	
	dbgout(SPrintf("Time: %f s. Image generation speed (img/s): %f\n. Localization speed (img/s): %f\n", tloc+tgen, N/tgen, N/tloc));
	dbgout(SPrintf("Average distance: COM x: %f, y: %f\n", comdist.x/N, comdist.y/N));
	dbgout(SPrintf("Average distance: Cross-correlation x: %f, y: %f\n", xcordist.x/N, xcordist.y/N));
}

void OnePixelTest()
{
	CPUTracker tracker(32,32, 16);

	tracker.getPixel(15,15) = 1;
	dbgout(SPrintf("Pixel at 15,15\n"));
	vector2f com = tracker.ComputeCOM(0);
	dbgout(SPrintf("COM: %f,%f\n", com.x, com.y));
	
	vector2f initial = {15,15};
	vector2f xcor = tracker.ComputeXCor(initial);
	dbgout(SPrintf("XCor: %f,%f\n", xcor.x, xcor.y));

	assert(xcor.x == 15.0f && xcor.y == 15.0f);
}

void SmallImageTest()
{
	CPUTracker tracker(32,32, 16);

	GenerateTestImage(&tracker, 15,15, 1, 0.0f);

	vector2f com = tracker.ComputeCOM(tracker.ComputeMedian());
	dbgout(SPrintf("COM: %f,%f\n", com.x, com.y));
	
	vector2f initial = {15,15};
	vector2f xcor = tracker.ComputeXCor(initial);
	dbgout(SPrintf("XCor: %f,%f\n", xcor.x, xcor.y));

	assert(fabsf(xcor.x-15.0f) < 1e-6 && fabsf(xcor.y-15.0f) < 1e-6);
}


void PixelationErrorTest()
{
	CPUTracker tracker(128,128, 64);

	float X = tracker.width/2;
	float Y = tracker.height/2;
	int N = 20;
	for (int x=0;x<N;x++)  {
		float xpos = X + 2.0f * x / (float)N;
		GenerateTestImage(&tracker, xpos, X, 1, 0.0f);

		vector2f com = tracker.ComputeCOM(tracker.ComputeMedian());
		//dbgout(SPrintf("COM: %f,%f\n", com.x, com.y));

		vector2f initial = {X,Y};
		vector2f xcor = tracker.ComputeXCor(initial);
		vector2f xcorInterp = tracker.ComputeXCorInterpolated(initial, 2);
		if (x==5) tracker.OutputDebugInfo();
		dbgout(SPrintf("xpos:%f, COM err: %f, XCor err: %f, XCorInterp err: %f\n", xpos, com.x-xpos, xcor.x-xpos, xcorInterp.x-xpos));
	}
}

int main()
{
//	SpeedTest();

	//SmallImageTest();
	PixelationErrorTest();

	return 0;
}

