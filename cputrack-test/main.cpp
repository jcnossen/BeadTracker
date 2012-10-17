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
	xcor = t->ComputeXCor(initial, 1);
}

int main()
{
	int N = 500;
	CPUTracker tracker(150,150, 128);

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

