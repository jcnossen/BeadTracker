
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

#include "simplefft.h"
#include "cudafft/cudafft.h"

double getPreciseTime()
{
	uint64_t freq, time;

	QueryPerformanceCounter((LARGE_INTEGER*)&time);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	return (double)time / (double)freq;
}


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

texture<float, cudaTextureType2D, cudaReadModeElementType> smpImgRef(0, cudaFilterModeLinear);


void TestSimpleFFT()
{
	int N=64;
	cudafft<double> fft(N, false);

	std::vector< cudafft<double>::cpx_type > data(N), result(N), cpu_result(N);
	for (int x=0;x<N;x++) {
		data[x].x = 10*cos(x*0.1f-5);
		data[x].y = 6*cos(x*0.2f-2)+3;
	}

	std::vector<sfft::complex<double> > twiddles = sfft::fill_twiddles<double>(N);

	fft.host_transform(&data[0], &cpu_result[0]);
	sfft::fft_forward(N, (sfft::complex<double>*)&data[0], &twiddles[0]);
	
	for (int k=0;k<N;k++) {
		dbgprintf("[%d] kissfft: %f+%fi, sfft: %f+%fi. diff=%f+%fi\n", k, cpu_result[k].x, cpu_result[k].y, data[k].x,data[k].y, cpu_result[k].x - data[k].x,cpu_result[k].y - data[k].y);
	}
}


void ShowCUDAError() {
	cudaError_t err = cudaGetLastError();
	dbgprintf("Cuda error: %s\n", cudaGetErrorString(err));
}

__global__ void testJobPassing(CUDATrackerJob job, CUDATrackerJob* a)
{
	CUDATrackerJob* a0 = &a[0];
	CUDATrackerJob* a1 = &a[1];
}

void TestJobPassing()
{
	CUDATrackerJob job;
	job.id = 1;
	job.initialPos.x = 2;
	job.initialPos.y = 3;
	job.initialPos.z = 4;
	job.zlut = 5;
	job.zlutPlane = 6;
	job.locType = LocalizeBuildZLUT;

	std::vector<CUDATrackerJob> jobs;
	jobs.push_back(job);
	jobs.push_back(job);

	testJobPassing<<<dim3(),dim3()>>>(job, device_vec<CUDATrackerJob> (jobs).data);
}

void TestLocalization()
{
	const int NumImages=480;
	int N = 10;
	QTrkSettings cfg;
	cfg.numThreads = -1;
	cfg.qi_iterations = 2;
	cfg.qi_maxradius = 30;
	QueuedCUDATracker trk(&cfg);

	auto images = cudaImageListf::alloc(80,80, NumImages, trk.UseHostEmulate());
	ShowCUDAError();
	std::vector<float3> positions(images.count);
	{
	device_vec< sfft::complex<float> > test;
	test = trk.DeviceMem( std::vector < sfft::complex<float> > (3) );
	}
	for(int i=0;i<images.count;i++) {
		float xp = images.w/2+(rand_uniform<float>() - 0.5) * 5;
		float yp = images.h/2+(rand_uniform<float>() - 0.5) * 5;
		positions[i] = make_float3(xp, yp, 3);
	}
	device_vec<float3> d_pos = trk.DeviceMem(positions);

	dbgprintf("Generating... %d images\n", N*images.count);

	double t0 = getPreciseTime();
	for (int i=0;i<N;i++)
		trk.GenerateImages(images, d_pos.data);
	cudaDeviceSynchronize();
	double tgen = getPreciseTime() - t0;

	auto d_com = trk.DeviceMem<float2>(positions.size());
	auto d_qi = trk.DeviceMem<float2>(positions.size());
	double t1 = getPreciseTime();
	dbgprintf("COM\n");
	for (int i=0;i<N;i++)
		trk.ComputeBgCorrectedCOM(images, d_com.data);
	cudaDeviceSynchronize();
	double t2 = getPreciseTime();
	double tcom = t2 - t1;

	dbgprintf("QI\n");
	for (int i=0;i<N;i++)
		trk.ComputeQI(images, d_com.data, d_qi.data);
	cudaDeviceSynchronize();
	double tqi = getPreciseTime() - t2;

	std::vector<float2> com(d_com), qi(d_qi);
	double comErrX=0, comErrY=0;
	double qiErrX=0, qiErrY=0;

	for (int i=0;i<images.count;i++) {
		dbgprintf("[%d] true pos=( %.4f, %.4f ).  COM error=( %.4f, %.4f ).  QI error=( %.4f, %.4f ) \n", i, 
			positions[i].x, positions[i].y, com[i].x - positions[i].x, com[i].y - positions[i].y, qi[i].x - positions[i].x, qi[i].y - positions[i].y );

		qiErrX += fabsf(positions[i].x-qi[i].x);
		qiErrY += fabsf(positions[i].y-qi[i].y);
		comErrX += fabsf(positions[i].x-com[i].x);
		comErrY += fabsf(positions[i].y-com[i].y);
	}

	dbgprintf("Errors: COM.x: %f, COM.y: %f, QI.x=%f, QI.y=%f\n", comErrX/images.count, comErrY/images.count, qiErrX/images.count, qiErrY/images.count);
	N *= images.count;
	dbgprintf("Image generating: %f img/s. COM: %f img/s. QI: %f img/s\n", N/tgen, N/tcom, N/tqi);

	ShowCUDAError();
	images.free();
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

	double t0 = getPreciseTime();
	testWithGlobal<<<nblocks,nthreads>>>(n,s,result_g.data,buf.data);
	cudaDeviceSynchronize();
	double t1 = getPreciseTime();
	testWithShared <<<nblocks,nthreads,s*sizeof(float)*nthreads.x>>>(n,s,result_s.data);
	cudaDeviceSynchronize();
	double t2 = getPreciseTime();

	std::vector<float> rs = result_s, rg = result_g;
	for (int x=0;x<n;x++) {
		dbgprintf("result_s[%d]=%f.   result_g[%d]=%f\n", x,rs[x], x,rg[x]);
	}

	dbgprintf("Speed of shared comp: %f, speed of global comp: %f\n", n/(t2-t1), n/(t1-t0));
}


void QTrkTest()
{
	QTrkSettings cfg;
	cfg.width = cfg.height = 60;
	cfg.qi_iterations = 4;
	cfg.qi_maxradius = 50;
	cfg.xc1_iterations = 2;
	cfg.xc1_profileLength = 64;
	cfg.numThreads = -1;
	cfg.com_bgcorrection = 0.0f;
	//cfg.numThreads = 6;
	QueuedCUDATracker qtrk(&cfg, -1);
	float *image = new float[cfg.width*cfg.height];

	// Generate ZLUT
	bool haveZLUT = true;
	int radialSteps=64, zplanes=100;
	float zmin=0.5,zmax=3;
	qtrk.SetZLUT(0, 1, zplanes, radialSteps);
	if (haveZLUT) {
		for (int x=0;x<zplanes;x++)  {
			vector2f center = { cfg.width/2, cfg.height/2 };
			float s = zmin + (zmax-zmin) * x/(float)(zplanes-1);
			GenerateTestImage(ImageData(image, cfg.width, cfg.height), center.x, center.y, s, 0.0f);
			uchar* zlutimg = floatToNormalizedInt(image, cfg.width,cfg.height, (uchar)255);
			WriteJPEGFile(zlutimg, cfg.width,cfg.height, "qtrkzlutimg.jpg", 99);
			delete[] zlutimg;
			qtrk.ScheduleLocalization((uchar*)image, cfg.width*sizeof(float),QTrkFloat, (LocalizeType)(LocalizeBuildZLUT|LocalizeOnlyCOM), x, 0, 0, x);
		}
		qtrk.Flush();
		// wait to finish ZLUT
		while(true) {
			int rc = qtrk.GetResultCount();
			if (rc == zplanes) break;
			Sleep(100);
			dbgprintf(".");
		}
	}
	float* zlut = qtrk.GetZLUT(0,0,0);
	qtrk.ClearResults();
	uchar* zlut_bytes = floatToNormalizedInt(zlut, radialSteps, zplanes, (uchar)255);
	WriteJPEGFile(zlut_bytes, radialSteps, zplanes, "qtrkzlutcuda.jpg", 99);
	delete[] zlut; delete[] zlut_bytes;
	
	// Schedule images to localize on
#ifdef _DEBUG
	int total= 1000;
#else
	int total = 50000;
#endif
	dbgprintf("Benchmarking...\n", total);
	GenerateTestImage(ImageData(image, cfg.width, cfg.height), cfg.width/2+1, cfg.height/2, zmin, 30);
	float maxv = image[0], minv =image[0];
	for (int x=0;x<cfg.width*cfg.height;x++) {
		maxv = std::max(maxv, image[x]);
		minv = std::min(minv, image[x]);
	}
	double tstart = getPreciseTime();
	int rc = 0, displayrc=0;
	for (int n=0;n<total;n++) {
		qtrk.ScheduleLocalization((uchar*)image, cfg.width*sizeof(float), QTrkFloat, (LocalizeType)(LocalizeQI|LocalizeZ), n, 0, 0, 0);
		if (n % 10 == 0) {
			rc = qtrk.GetResultCount();
			while (displayrc<rc) {
				if( displayrc%(total/10)==0) dbgprintf("Done: %d / %d\n", displayrc, total);
				displayrc++;
			}
		}
	}
	qtrk.Flush();
	do {
		rc = qtrk.GetResultCount();
		while (displayrc<rc) {
			if( displayrc%(total/10)==0) dbgprintf("Done: %d / %d\n", displayrc, total);
			displayrc++;
		}
		Sleep(10);
	} while (rc != total);
	
	// Measure speed
	double tend = getPreciseTime();

	delete[] image;

	LocalizationResult r;
	qtrk.PollFinished(&r, 1);
	dbgprintf("Result.x: %f, Result.y: %f\n", r.pos.x, r.pos.y);

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

int main(int argc, char *argv[])
{
//	testLinearArray();

	//TestJobPassing();
	//TestLocalization();
	//TestSimpleFFT();
	//TestKernelFFT();
//	TestSharedMem();
	QTrkTest();

	listDevices();
	return 0;
}
