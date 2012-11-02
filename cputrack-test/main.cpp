#include "../cputrack/cpu_tracker.h"
#include <Windows.h>
#include <stdint.h>
#include "../cputrack/random_distr.h"
#include "../cputrack/FFT2DTracker.h" 
#include "../cputrack/queued_cpu_tracker.h"

template<typename T> T sq(T x) { return x*x; }
template<typename T> T distance(T x, T y) { return sqrt(x*x+y*y); }


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
	CPUTracker* tracker = new CPUTracker(150,150, 64);

	int radialSteps = 64, zplanes = 120;
	float* zlut = new float[radialSteps*zplanes];
	float zmin = 2, zmax = 8;
	float zradius = tracker->xcorw/2;

	for (int x=0;x<zplanes;x++)  {
		vector2f center = { tracker->GetWidth()/2, tracker->GetHeight()/2 };
		float s = zmin + (zmax-zmin) * x/(float)(zplanes-1);
		GenerateTestImage(tracker->srcImage, tracker->GetWidth(), tracker->GetHeight(), center.x, center.y, s, 0.0f);
		tracker->ComputeRadialProfile(&zlut[x*radialSteps], radialSteps, 64, zradius, center);
	}
	tracker->SetZLUT(zlut, zplanes, radialSteps, 1, zradius, 64, true);
	delete[] zlut;

	// Speed test
	vector2f comdist={}, xcordist={};
	float zdist=0.0f;
	double zerrsum=0.0f;
	double tloc = 0.0, tgen=0.0, tz = 0.0;
	for (int k=0;k<N;k++)
	{
		double t0 = getPreciseTime();
		float xp = tracker->GetWidth()/2+(rand_uniform<float>() - 0.5) * 5;
		float yp = tracker->GetHeight()/2+(rand_uniform<float>() - 0.5) * 5;
		float z = zmin + 0.1f + (zmax-zmin-0.2f) * rand_uniform<float>();

		GenerateTestImage(tracker->srcImage, tracker->GetWidth(), tracker->GetHeight(), xp, yp, z, 10000);

		double t1 = getPreciseTime();
		vector2f com = tracker->ComputeBgCorrectedCOM();
		vector2f initial = {com.x, com.y};
		vector2f xcor = tracker->ComputeXCorInterpolated(initial, 2);
/*		if (k == 1) {
			tracker.OutputDebugInfo();
			writeImageAsCSV("test.csv", tracker.srcImage, tracker.width, tracker.height);
		}*/

		comdist.x += fabsf(com.x - xp);
		comdist.y += fabsf(com.y - yp);

		xcordist.x += fabsf(xcor.x - xp);
		xcordist.y += fabsf(xcor.y - yp);
		double t2 = getPreciseTime();

		float est_z = zmin + tracker->ComputeZ(xcor, 64, 0) * (zmax - zmin);
		zdist += fabsf(est_z-z);
		zerrsum += est_z-z;

		double t3 = getPreciseTime();
	//	dbgout(SPrintf("xpos:%f, COM err: %f, XCor err: %f\n", xp, com.x-xp, xcor.x-xp));
		if (k>0) { // skip first initialization round
			tloc+=t2-t1;
			tgen+=t1-t0;
			tz+=t3-t2;
		}
	}

	int Nns = N-1;
	dbgprintf("Time: %f s. Image gen. (img/s): %f\n2D loc. speed (img/s): %f Z estimation (img/s): %f\n", tloc+tgen, Nns/tgen, Nns/tloc, Nns/tz);
	dbgprintf("Average dist: COM x: %f, y: %f\n", comdist.x/N, comdist.y/N);
	dbgprintf("Average dist: Cross-correlation x: %f, y: %f\n", xcordist.x/N, xcordist.y/N);
	dbgprintf("Average dist: Z: %f. Mean error:%f\n", zdist/N, zerrsum/N); 
	
	delete tracker;
}

void OnePixelTest()
{
	CPUTracker* tracker = new CPUTracker(32,32, 16);

	tracker->getPixel(15,15) = 1;
	dbgout(SPrintf("Pixel at 15,15\n"));
	vector2f com = tracker->ComputeBgCorrectedCOM();
	dbgout(SPrintf("COM: %f,%f\n", com.x, com.y));
	
	vector2f initial = {15,15};
	vector2f xcor = tracker->ComputeXCor(initial);
	dbgout(SPrintf("XCor: %f,%f\n", xcor.x, xcor.y));

	assert(xcor.x == 15.0f && xcor.y == 15.0f);
	delete tracker;
}
 
void SmallImageTest()
{
	CPUTracker *tracker = new CPUTracker(32,32, 16);

	GenerateTestImage(tracker->srcImage, tracker->GetWidth(), tracker->GetHeight(), 15,15, 1, 0.0f);

	vector2f com = tracker->ComputeBgCorrectedCOM();
	dbgout(SPrintf("COM: %f,%f\n", com.x, com.y));
	
	vector2f initial = {15,15};
	vector2f xcor = tracker->ComputeXCor(initial);
	dbgout(SPrintf("XCor: %f,%f\n", xcor.x, xcor.y));

	assert(fabsf(xcor.x-15.0f) < 1e-6 && fabsf(xcor.y-15.0f) < 1e-6);
	delete tracker;
}


void PixelationErrorTest()
{
	CPUTracker *tracker = new CPUTracker(128,128, 64);

	float X = tracker->GetWidth()/2;
	float Y = tracker->GetHeight()/2;
	int N = 20;
	for (int x=0;x<N;x++)  {
		float xpos = X + 2.0f * x / (float)N;
		GenerateTestImage(tracker->srcImage, tracker->GetWidth(), tracker->GetHeight(), xpos, X, 1, 0.0f);

		vector2f com = tracker->ComputeBgCorrectedCOM();
		//dbgout(SPrintf("COM: %f,%f\n", com.x, com.y));

		vector2f initial = {X,Y};
		vector2f xcor = tracker->ComputeXCor(initial);
		vector2f xcorInterp = tracker->ComputeXCorInterpolated(initial, 3);
		dbgout(SPrintf("xpos:%f, COM err: %f, XCor err: %f, XCorInterp err: %f\n", xpos, com.x-xpos, xcor.x-xpos, xcorInterp.x-xpos));
	}
	delete tracker;
}

float EstimateZError(int zplanes)
{
	// build LUT
	CPUTracker *tracker = new CPUTracker(128,128, 64);

	vector2f center = { tracker->GetWidth()/2, tracker->GetHeight()/2 };
	int radialSteps = 64;
	float* zlut = new float[radialSteps*zplanes];
	float zmin = 2, zmax = 8;
	float zradius = tracker->xcorw/2;

	//GenerateTestImage(&tracker, center.x, center.y, zmin, 0.0f);
	//writeImageAsCSV("img.csv", tracker.srcImage, tracker.width, tracker.height);

	for (int x=0;x<zplanes;x++)  {
		float s = zmin + (zmax-zmin) * x/(float)(zplanes-1);
		GenerateTestImage(tracker->srcImage, tracker->GetWidth(), tracker->GetHeight(), center.x, center.y, s, 0.0f);
	//	dbgout(SPrintf("z=%f\n", s));
		tracker->ComputeRadialProfile(&zlut[x*radialSteps], radialSteps, 64, zradius, center);
	}

	tracker->SetZLUT(zlut, zplanes, radialSteps, 1, zradius, 64, true);
	writeImageAsCSV("zlut.csv", zlut, radialSteps, zplanes);
	delete[] zlut;

	int N=100;
	float zdist=0.0f;
	for (int k=0;k<N;k++) {
		float z = zmin + k/float(N-1) * (zmax-zmin);
		GenerateTestImage(tracker->srcImage, tracker->GetWidth(), tracker->GetHeight(), center.x, center.y, z, 0.0f);
		
		float est_z = zmin + tracker->ComputeZ(center, 64, 0) * (zmax - zmin);
		zdist += fabsf(est_z-z);
		//dbgout(SPrintf("Z: %f, EstZ: %f\n", z, est_z));

		if(k==50) {
			writeImageAsCSV("rprofdiff.csv", &tracker->rprof_diff[0], tracker->rprof_diff.size(),1);
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

/*
void Test2DTracking()
{
	CPUTracker tracker(150,150);

	float zmin = 2;
	float zmax = 6;
	int N = 200;

	double tloc2D = 0, tloc1D = 0;
	double dist2D = 0;
	double dist1D = 0;
	for (int k=0;k<N;k++) {
		float xp = tracker.GetWidth()/2+(rand_uniform<float>() - 0.5) * 5;
		float yp = tracker.GetHeight()/2+(rand_uniform<float>() - 0.5) * 5;
		float z = zmin + 0.1f + (zmax-zmin-0.2f) * rand_uniform<float>();

		GenerateTestImage(tracker.srcImage, tracker.GetWidth(), tracker.GetHeight(), xp, yp, z, 50000);

		double t0 = getPreciseTime();
		vector2f xcor2D = tracker.ComputeXCor2D();
		if (k==0) {
			float * results = tracker.tracker2D->GetAutoConvResults();
			writeImageAsCSV("xcor2d-autoconv-img.csv", results, tracker.GetWidth(), tracker.GetHeight());
		}

		double t1 = getPreciseTime();
		vector2f com = tracker.ComputeBgCorrectedCOM();
		vector2f xcor1D = tracker.ComputeXCorInterpolated(com, 2);
		double t2 = getPreciseTime();

		dist1D += distance(xp-xcor1D.x,yp-xcor1D.y);
		dist2D += distance(xp-xcor2D.x,yp-xcor2D.y);

		if (k>0) {
			tloc2D += t1-t0;
			tloc1D += t2-t1;
		}
	}
	N--; // ignore first

	dbgprintf("1D Xcor speed(img/s): %f\n2D Xcor speed (img/s): %f\n", N/tloc1D, N/tloc2D);
	dbgprintf("Average dist XCor 1D: %f\n", dist1D/N);
	dbgprintf("Average dist XCor 2D: %f\n", dist2D/N);
}*/

void QTrkTest()
{
	QTrkSettings cfg;
	QueuedCPUTracker qtrk(&cfg);

	int NumImages=10, JobsPerImg=1000;
	dbgprintf("Generating %d images...\n", NumImages);
	float *image = new float[cfg.width*cfg.height];
	float zmin = 2.0f, zmax=6.0f;
	double tgen = 0.0, tschedule = 0.0;
	for (int n=0;n<NumImages;n++) {
		double t1 = getPreciseTime();
		float xp = cfg.width/2+(rand_uniform<float>() - 0.5) * 5;
		float yp = cfg.height/2+(rand_uniform<float>() - 0.5) * 5;
		float z = zmin + 0.1f + (zmax-zmin-0.2f) * rand_uniform<float>();

		GenerateTestImage(image, cfg.width, cfg.height, xp, yp, z, 10000);
		double t2 = getPreciseTime();
		for (int k=0;k<JobsPerImg;k++)
			qtrk.ScheduleLocalization((uchar*)image, cfg.width*sizeof(float), QTrkFloat, LocalizeXCor1D, false, n);
		double t3 = getPreciseTime();
		tgen += t2-t1;
		tschedule += t3-t2;
	}
	delete[] image;
	dbgprintf("Schedule time: %f, Generation time: %f\n", tschedule, tgen);

	dbgprintf("Localizing on %d images...\n", NumImages*JobsPerImg);

	double tstart = getPreciseTime();
	int jobc = 0;
	int hjobc = qtrk.JobCount();
	int startJobs = hjobc;
	qtrk.Start();
	do {
		jobc = qtrk.JobCount();
		while (hjobc>jobc) {
			if( hjobc%100==0) dbgprintf("TODO: %d\n", hjobc);
			hjobc--;
		}
		Sleep(5);
	} while (jobc!=0);
	double tend = getPreciseTime();
	dbgprintf("Localization Speed: %d (img/s), using %d threads\n", (int)( startJobs/(tend-tstart) ), qtrk.NumThreads());
}

int main()
{
	//SpeedTest();
	//SmallImageTest();
	//PixelationErrorTest();
	//ZTrackingTest();
	//Test2DTracking();
	QTrkTest();

	return 0;
}
