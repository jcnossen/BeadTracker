
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

void TestLocalization()
{
#ifdef _DEBUG
	const int NumImages=4;
#else
	const int NumImages=256;
#endif
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
	/*
	for (int i=0;i<images.count;i++) {
		dbgprintf("[%d] true pos=( %.4f, %.4f ).  COM error=( %.4f, %.4f ).  QI error=( %.4f, %.4f ) \n", i, 
			positions[i].x, positions[i].y, com[i].x - positions[i].x, com[i].y - positions[i].y, qi[i].x - positions[i].x, qi[i].y - positions[i].y );
	}*/

	N *= images.count;
	dbgprintf("Image generating: %f img/s. COM: %f img/s. QI: %f img/s\n", N/tgen, N/tcom, N/tqi);

	ShowCUDAError();
	images.free();
}

int main(int argc, char *argv[])
{
//	testLinearArray();

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);

	std::string path = getPath(argv[0]);

	TestLocalization();
	//TestSimpleFFT();
	//TestKernelFFT();

	return 0;
}
