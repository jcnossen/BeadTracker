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


void writeImageAsCSV(const char* file, float* d, int w,int h)
{
	FILE* f = fopen(file, "w");

	for (int y=0;y<h;y++) {
		for (int x=0;x<w;x++)
		{
			fprintf(f, "%f", d[y*w+x]);
			if(x<w-1) fputs("\t", f); 
		}
		fprintf(f, "\n");
	}

	fclose(f);
}

void SpeedTest()
{
	int N = 1000;
	CPUTracker tracker(150,150, 64);

	int radialSteps = 64, zplanes = 120;
	float* zlut = new float[radialSteps*zplanes];
	float zmin = 2, zmax = 8;
	float zradius = tracker.xcorw/2;

	for (int x=0;x<zplanes;x++)  {
		vector2f center = { tracker.width/2, tracker.height/2 };
		float s = zmin + (zmax-zmin) * x/(float)(zplanes-1);
		GenerateTestImage(&tracker, center.x, center.y, s, 0.0f);
		tracker.ComputeRadialProfile(&zlut[x*radialSteps], radialSteps, 64, zradius, center);
	}
	tracker.SetZLUT(zlut, zplanes, radialSteps);

	// Speed test
	vector2f comdist={}, xcordist={};
	float zdist=0.0f;
	double tloc = 0.0, tgen=0.0, tz = 0.0;
	for (int k=0;k<N;k++)
	{
		double t0= getPreciseTime();
		float xp = tracker.width/2+(rand_uniform<float>() - 0.5) * 5;
		float yp = tracker.height/2+(rand_uniform<float>() - 0.5) * 5;
		float z = zmin + 0.1f + (zmax-zmin-0.2f) * rand_uniform<float>();

		GenerateTestImage(&tracker, xp, yp, z, 0);

		double t1 = getPreciseTime();
		float median = tracker.ComputeMedian();
		vector2f com = tracker.ComputeCOM(median);
		vector2f initial = {com.x, com.y};
		vector2f xcor = tracker.ComputeXCorInterpolated(initial,2);
/*		if (k == 1) {
			tracker.OutputDebugInfo();
			writeImageAsCSV("test.csv", tracker.srcImage, tracker.width, tracker.height);
		}*/

		comdist.x += fabsf(com.x - xp);
		comdist.y += fabsf(com.y - yp);

		xcordist.x += fabsf(xcor.x - xp);
		xcordist.y += fabsf(xcor.y - yp);
		double t2 = getPreciseTime();

		float est_z = zmin + tracker.ComputeZ(xcor, 64, zradius) * (zmax - zmin) / (zplanes - 1);
		zdist += fabsf(est_z-z);

		double t3 = getPreciseTime();
	//	dbgout(SPrintf("xpos:%f, COM err: %f, XCor err: %f\n", xp, com.x-xp, xcor.x-xp));
		tloc+=t2-t1;
		tgen+=t1-t0;
		tz+=t3-t2;
	}
	
	dbgout(SPrintf("Time: %f s. Image gen. (img/s): %f\n2D loc. speed (img/s): %f Z estimation (img/s): %f\n", tloc+tgen, N/tgen, N/tloc, N/tz));
	dbgout(SPrintf("Average dist: COM x: %f, y: %f\n", comdist.x/N, comdist.y/N));
	dbgout(SPrintf("Average dist: Cross-correlation x: %f, y: %f\n", xcordist.x/N, xcordist.y/N));
	dbgout(SPrintf("Average dist: Z: %f\n", zdist/N)); 
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
		vector2f xcorInterp = tracker.ComputeXCorInterpolated(initial, 3);
		dbgout(SPrintf("xpos:%f, COM err: %f, XCor err: %f, XCorInterp err: %f\n", xpos, com.x-xpos, xcor.x-xpos, xcorInterp.x-xpos));
	}
}

float EstimateZError(int zplanes)
{
	// build LUT
	CPUTracker tracker(128,128, 64);

	vector2f center = { tracker.width/2, tracker.height/2 };
	int radialSteps = 64;
	float* zlut = new float[radialSteps*zplanes];
	float zmin = 2, zmax = 8;
	float zradius = tracker.xcorw/2;

	//GenerateTestImage(&tracker, center.x, center.y, zmin, 0.0f);
	//writeImageAsCSV("img.csv", tracker.srcImage, tracker.width, tracker.height);

	for (int x=0;x<zplanes;x++)  {
		float s = zmin + (zmax-zmin) * x/(float)(zplanes-1);
		GenerateTestImage(&tracker, center.x, center.y, s, 0.0f);
	//	dbgout(SPrintf("z=%f\n", s));
		tracker.ComputeRadialProfile(&zlut[x*radialSteps], radialSteps, 64, zradius, center);
	}

	tracker.SetZLUT(zlut, zplanes, radialSteps);
	writeImageAsCSV("zlut.csv", zlut, radialSteps, zplanes);

	int N=100;
	float zdist=0.0f;
	for (int k=0;k<N;k++) {
		float z = zmin + k/float(N-1) * (zmax-zmin);
		GenerateTestImage(&tracker, center.x, center.y, z, 0.0f);
		
		float est_z = zmin + tracker.ComputeZ(center, 64, zradius) * (zmax - zmin) / (zplanes - 1);
		zdist += fabsf(est_z-z);
		//dbgout(SPrintf("Z: %f, EstZ: %f\n", z, est_z));

		if(k==50) {
			writeImageAsCSV("rprofdiff.csv", &tracker.rprof_diff[0], tracker.rprof_diff.size(),1);
		}
	}
	return zdist/N;
}


void ZTrackingTest()
{
	for (int k=20;k<100;k+=10)
	{
		float err = EstimateZError(k);
		dbgout(SPrintf("average Z difference: %f. zplanes=%d\n", err, k));
	}
}

int main()
{
	SpeedTest();

	//SmallImageTest();
	//PixelationErrorTest();
//	ZTrackingTest();

	return 0;
}

