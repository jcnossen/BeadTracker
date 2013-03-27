
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "std_incl.h"
#include "utils.h"

#include <cassert>
#include <cstdlib>
#include <stdio.h>
#include <windows.h>
#include <cstdarg>
#include <valarray>

#include "random_distr.h"

#include <stdint.h>
#include "gpu_utils.h"
#include "QueuedCUDATracker.h"
#include "queued_cpu_tracker.h"



std::string getPath(const char *file)
{
	std::string s = file;
	int pos = s.length()-1;
	while (pos>0 && s[pos]!='\\' && s[pos]!= '/' )
		pos--;
	
	return s.substr(0, pos);
}


inline __device__ float2 mul_conjugate(float2 a, float2 b)
{
	float2 r;
	r.x = a.x*b.x + a.y*b.y;
	r.y = a.y*b.x - a.x*b.y;
	return r;
}


void ShowCUDAError() {
	cudaError_t err = cudaGetLastError();
	dbgprintf("Cuda error: %s\n", cudaGetErrorString(err));
}

__shared__ float cudaSharedMem[];

__device__ float compute(int idx, float* buf, int s)
{
	// some random calcs to make the kernel unempty
	float k=0.0f;
	for (int x=0;x<s;x++ ){
		k+=cosf(x*0.1f*idx);
		buf[x]=k;
	}
	for (int x=0;x<s/2;x++){
		buf[x]=buf[x]*buf[x];
	}
	float sum=0.0f;
	for (int x=s-1;x>=1;x--) {
		sum += buf[x-1]/(fabsf(buf[x])+0.1f);
	}
	return sum;
}


__global__ void testWithGlobal(int n, int s, float* result, float* buf) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		result [idx] = compute(idx, &buf [idx * s],s);
	}
}

__global__ void testWithShared(int n, int s, float* result) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		result [idx] = compute(idx, &cudaSharedMem[threadIdx.x * s],s);
	}
}

void TestSharedMem()
{
	int n=100, s=200;
	dim3 nthreads(32), nblocks( (n+nthreads.x-1)/nthreads.x);
	device_vec<float> buf(n*s);
	device_vec<float> result_s(n), result_g(n);

	double t0 = GetPreciseTime();
	testWithGlobal<<<nblocks,nthreads>>>(n,s,result_g.data,buf.data);
	cudaDeviceSynchronize();
	double t1 = GetPreciseTime();
	testWithShared <<<nblocks,nthreads,s*sizeof(float)*nthreads.x>>>(n,s,result_s.data);
	cudaDeviceSynchronize();
	double t2 = GetPreciseTime();

	std::vector<float> rs = result_s, rg = result_g;
	for (int x=0;x<n;x++) {
		dbgprintf("result_s[%d]=%f.   result_g[%d]=%f\n", x,rs[x], x,rg[x]);
	}

	dbgprintf("Speed of shared comp: %f, speed of global comp: %f\n", n/(t2-t1), n/(t1-t0));
}


void QTrkTest()
{
	QTrkSettings cfg;
	cfg.width = cfg.height = 120;
	cfg.qi_iterations = 1;
	cfg.qi_maxradius = 25;
	cfg.xc1_iterations = 2;
	cfg.xc1_profileLength = 64;
	cfg.numThreads = -1;
	cfg.com_bgcorrection = 0.0f;
	cfg.zlut_maxradius = 30;
	cfg.zlut_radialsteps = 64;
	cfg.zlut_angularsteps = 128;
	bool haveZLUT = false;
#ifdef _DEBUG
	cfg.qi_radialsteps=16;
	cfg.numThreads = 2;
	cfg.qi_iterations=1;
	int total= 1;
	int batchSize = 1;
	haveZLUT=true;
#else
	cfg.numThreads = 4;
	int total = 30000;
	int batchSize = 512;
#endif

	QueuedCUDATracker qtrk(&cfg, batchSize);
	QueuedCPUTracker qtrkcpu(&cfg);
	float *image = new float[cfg.width*cfg.height];
	bool cpucmp = true;

	srand(1);

	// Generate ZLUT
	int zplanes=100;
	float zmin=0.5,zmax=3;
	qtrk.SetZLUT(0, 1, zplanes);
	if (cpucmp) qtrkcpu.SetZLUT(0, 1, zplanes);
	if (haveZLUT) {
		for (int x=0;x<zplanes;x++)  {
			vector2f center = { cfg.width/2, cfg.height/2 };
			float s = zmin + (zmax-zmin) * x/(float)(zplanes-1);
			GenerateTestImage(ImageData(image, cfg.width, cfg.height), center.x, center.y, s, 0.0f);
			uchar* zlutimg = floatToNormalizedInt(image, cfg.width,cfg.height, (uchar)255);
			WriteJPEGFile(zlutimg, cfg.width,cfg.height, "qtrkzlutimg.jpg", 99);
			delete[] zlutimg;
			LocalizeType flags = (LocalizeType)(LocalizeBuildZLUT|LocalizeOnlyCOM);
			qtrk.ScheduleLocalization((uchar*)image, cfg.width*sizeof(float),QTrkFloat,flags , x, 0, 0, x);
			if (cpucmp) qtrkcpu.ScheduleLocalization((uchar*)image, cfg.width*sizeof(float),QTrkFloat, flags, x, 0, 0, x);
		}
		qtrk.Flush();
		if (cpucmp) qtrkcpu.Flush();
		// wait to finish ZLUT
		while(true) {
			int rc = qtrk.GetResultCount();
			if (rc == zplanes) break;
			Sleep(100);
			dbgprintf(".");
		}
		if (cpucmp) {
			while(qtrkcpu.GetResultCount() != zplanes);
		}
	}
	float* zlut = qtrk.GetZLUT(0,0);
	if (cpucmp) { 
		float* zlutcpu = qtrkcpu.GetZLUT(0,0);

		WriteImageAsCSV("zlut-cpu.txt", zlutcpu, cfg.zlut_radialsteps, zplanes);
		WriteImageAsCSV("zlut-gpu.txt", zlut, cfg.zlut_radialsteps, zplanes);
	}
	qtrk.ClearResults();
	if (cpucmp) qtrkcpu.ClearResults();
	uchar* zlut_bytes = floatToNormalizedInt(zlut, cfg.zlut_radialsteps, zplanes, (uchar)255);
	WriteJPEGFile(zlut_bytes, cfg.zlut_radialsteps, zplanes, "qtrkzlutcuda.jpg", 99);
	delete[] zlut; delete[] zlut_bytes;
	
	// Schedule images to localize on
	dbgprintf("Benchmarking...\n", total);
	GenerateTestImage(ImageData(image, cfg.width, cfg.height), cfg.width/2, cfg.height/2, (zmin+zmax)/2, 0);
	double tstart = GetPreciseTime();
	int rc = 0, displayrc=0;
	for (int n=0;n<total;n++) {
		LocalizeType flags = (LocalizeType)(LocalizeQI| (haveZLUT ? LocalizeZ : 0) );
		qtrk.ScheduleLocalization((uchar*)image, cfg.width*sizeof(float), QTrkFloat, flags, n, 0, 0, 0);
		if (cpucmp) qtrkcpu.ScheduleLocalization((uchar*)image, cfg.width*sizeof(float), QTrkFloat, flags, n, 0, 0, 0);
		if (n % 10 == 0) {
			rc = qtrk.GetResultCount();
			while (displayrc<rc) {
				if( displayrc%(total/10)==0) dbgprintf("Done: %d / %d\n", displayrc, total);
				displayrc++;
			}
		}
	}
	if (cpucmp) qtrkcpu.Flush();
	qtrk.Flush();
	do {
		rc = qtrk.GetResultCount();
		while (displayrc<rc) {
			if( displayrc%std::max(1,total/10)==0) dbgprintf("Done: %d / %d\n", displayrc, total);
			displayrc++;
		}
		Sleep(10);
	} while (rc != total);
	
	// Measure speed
	double tend = GetPreciseTime();

	if (cpucmp) {
		dbgprintf("waiting for cpu results..\n");
		while (total != qtrkcpu.GetResultCount())
			Sleep(10);
	}
	

	delete[] image;

	const int NumResults = 20;
	LocalizationResult results[NumResults], resultscpu[NumResults];
	int rcount = std::min(NumResults,total);
	for (int i=0;i<rcount;i++) {
		qtrk.PollFinished(&results[i], 1);
		if (cpucmp) qtrkcpu.PollFinished(&resultscpu[i], 1);
	}
	std::sort(results, results+rcount, [](LocalizationResult a, LocalizationResult b) -> bool { return a.id > b.id; });
	if(cpucmp) std::sort(resultscpu, resultscpu+rcount, [](LocalizationResult a, LocalizationResult b) -> bool { return a.id > b.id; });
	for (int i=0;i<rcount;i++) {
		LocalizationResult& r = results[i];
		dbgprintf("gpu [%d] x: %f, y: %f. z: %+g, COM: %f, %f\n", i,r.pos.x, r.pos.y, r.z, r.firstGuess.x, r.firstGuess.y);

		if (cpucmp) {
			r = resultscpu[i];
			dbgprintf("cpu [%d] x: %f, y: %f. z: %+g, COM: %f, %f\n", i,r.pos.x, r.pos.y, r.z, r.firstGuess.x, r.firstGuess.y);
		}
	}

	dbgprintf("Localization Speed: %d (img/s)\n", (int)( total/(tend-tstart) ));
}

void listDevices()
{
	cudaDeviceProp prop;
	int dc;
	cudaGetDeviceCount(&dc);
	for (int k=0;k<dc;k++) {
		cudaGetDeviceProperties(&prop, k);
		dbgprintf("Device[%d] = %s\n", k, prop.name);
	}

}

__global__ void SimpleKernel(int N, float* a){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		for (int x=0;x<1000;x++)
			a[idx] = asin(a[idx]+x);
	}
}


void TestAsync()
{
	int N =100000;
	int nt = 32;

	pinned_array<float> a(N); 
//	cudaMallocHost(&a, sizeof(float)*N, 0);

	device_vec<float> A(N);

	cudaStream_t s0, s1;
	cudaEvent_t done;

	cudaStreamCreate(&s0);
	cudaEventCreate(&done,0);

	for (int x=0;x<N;x++)
		a[x] = cos(x*0.01f);

	for (int x=0;x<1;x++) {
		{ MeasureTime mt("a->A"); A.copyToDevice(a.data(), N, true); }
		{ MeasureTime mt("func(A)"); 
		SimpleKernel<<<dim3( (N+nt-1)/nt ), dim3(nt)>>>(N, A.data);
		}
		{ MeasureTime mt("A->a"); A.copyToHost(a.data(), true); }
	}
	cudaEventRecord(done);

	{
	MeasureTime("sync..."); while (cudaEventQuery(done) != cudaSuccess); 
	}
	
	cudaStreamDestroy(s0);
	cudaEventDestroy(done);
}

__global__ void emptyKernel()
{}

float SpeedTest(const QTrkSettings& cfg, QueuedTracker* qtrk, int count, bool haveZLUT, LocalizeType locType)
{
	float *image = new float[cfg.width*cfg.height];
	srand(1);

	// Generate ZLUT
	int zplanes=100;
	float zmin=0.5,zmax=3;
	qtrk->SetZLUT(0, 1, zplanes);
	if (haveZLUT) {
		for (int x=0;x<zplanes;x++)  {
			vector2f center = { cfg.width/2, cfg.height/2 };
			float s = zmin + (zmax-zmin) * x/(float)(zplanes-1);
			GenerateTestImage(ImageData(image, cfg.width, cfg.height), center.x, center.y, s, 0.0f);
			LocalizeType flags = (LocalizeType)(LocalizeBuildZLUT|LocalizeOnlyCOM);
			qtrk->ScheduleLocalization((uchar*)image, cfg.width*sizeof(float),QTrkFloat,flags , x, 0, 0, x);
		}
		qtrk->Flush();
		// wait to finish ZLUT
		while(true) {
			int rc = qtrk->GetResultCount();
			if (rc == zplanes) break;
			Sleep(100);
			dbgprintf(".");
		}
	}
	qtrk->ClearResults();
	
	// Schedule images to localize on
	dbgprintf("Benchmarking...\n", count);
	GenerateTestImage(ImageData(image, cfg.width, cfg.height), cfg.width/2, cfg.height/2, (zmin+zmax)/2, 0);
	double tstart = GetPreciseTime();
	int rc = 0, displayrc=0;
	for (int n=0;n<count;n++) {
		LocalizeType flags = (LocalizeType)(locType| (haveZLUT ? LocalizeZ : 0) );
		qtrk->ScheduleLocalization((uchar*)image, cfg.width*sizeof(float), QTrkFloat, flags, n, 0, 0, 0);
		if (n % 10 == 0) {
			rc = qtrk->GetResultCount();
			while (displayrc<rc) {
				if( displayrc%(count/10)==0) dbgprintf("Done: %d / %d\n", displayrc, count);
				displayrc++;
			}
		}
	}
	qtrk->Flush();
	do {
		rc = qtrk->GetResultCount();
		while (displayrc<rc) {
			if( displayrc%std::max(1,count/10)==0) dbgprintf("Done: %d / %d\n", displayrc, count);
			displayrc++;
		}
		Sleep(10);
	} while (rc != count);
	
	// Measure speed
	double tend = GetPreciseTime();
	delete[] image;

	return count/(tend-tstart);
}


void SpeedCompareTest()
{
	int count = 20000;
	bool haveZLUT = false;
	LocalizeType locType = LocalizeQI;

	QTrkSettings cfg;
	cfg.width = cfg.height = 120;
	cfg.qi_iterations = 1;
	cfg.qi_maxradius = 40;
	cfg.qi_angsteps_per_quadrant = 32;
	cfg.qi_radialsteps = 32;
	cfg.numThreads = -1;
	cfg.com_bgcorrection = 0.0f;
	cfg.zlut_maxradius = 40;
	cfg.zlut_radialsteps = 64;
	cfg.zlut_angularsteps = 128;

	QueuedCPUTracker *cputrk = new QueuedCPUTracker(&cfg);
	float cpuspeed = SpeedTest(cfg, cputrk, count, haveZLUT, locType);
	delete cputrk;

	QueuedCUDATracker *cudatrk = new QueuedCUDATracker(&cfg, 256);
	float gpuspeed = SpeedTest(cfg, cudatrk, count, haveZLUT, locType);
	delete cudatrk;

	auto profiling = QueuedCUDATracker::GetProfilingResults();
	for (auto i = profiling.begin(); i != profiling.end(); ++i) {
		auto r = i->second;
		dbgprintf("%s took %f ms on average\n", i->first, 1000*r.second/r.first);
	}

	dbgprintf("CPU tracking speed: %d img/s\n", (int)cpuspeed);
	dbgprintf("GPU tracking speed: %d img/s\n", (int)gpuspeed);
}


int main(int argc, char *argv[])
{
//	testLinearArray();
	
	SpeedCompareTest();

	listDevices();
	return 0;
}
