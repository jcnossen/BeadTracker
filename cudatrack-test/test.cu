
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


texture<float, cudaTextureType2D, cudaReadModeElementType> xcor1D_images(0, cudaFilterModeLinear);

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

	fft.host_transform(&data[0], &cpu_result[0]);
	sfft::fft_forward(N, (sfft::complex<double>*)&data[0]);
	
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
	QTrkSettings cfg;
	cfg.numThreads = -1;
	cfg.qi_iterations = 4;
	QueuedCUDATracker trk(&cfg);

	auto images = cudaImageListf::alloc(128,128,32, trk.UseHostEmulate());
	std::vector<float3> positions(images.count);

	for(int i=0;i<images.count;i++) {
		float xp = images.w/2+(rand_uniform<float>() - 0.5) * 5;
		float yp = images.h/2+(rand_uniform<float>() - 0.5) * 5;
		positions[i] = make_float3(xp, yp, 3);
	}
	trk.GenerateImages(images, trk.DeviceMem(positions).data);

	auto d_com = trk.DeviceMem<float2>(positions.size());
	auto d_qi = trk.DeviceMem<float2>(positions.size());
	trk.ComputeBgCorrectedCOM(images, d_com.data);
	trk.ComputeQI(images, d_com.data, d_qi.data);

	std::vector<float2> com(d_com), qi(d_qi);
	for (int i=0;i<images.count;i++) {
		dbgprintf("[%d] true pos=( %.4f, %.4f ).  COM error=( %.4f, %.4f ).  QI error=( %.4f, %.4f ) \n", i, 
			positions[i].x, positions[i].y, com[i].x - positions[i].x, com[i].y - positions[i].y, qi[i].x - positions[i].x, qi[i].y - positions[i].y );
	}

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
