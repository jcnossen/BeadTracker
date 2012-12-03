
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

#include "cudafft/cudafft.h"
#include "random_distr.h"

#include <stdint.h>
#include "cudaImageList.h"
#include "QueuedCUDATracker.h"

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


template<typename T>
__global__ void XCor1D_BuildProfiles_Kernel(cudaImageList<T> list, float2* centers, T* profiles, int profileLen, texture<float, cudaTextureType2D, cudaReadModeElementType> tex)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	T* hprof = &profiles[idx*profileLen*2];
	T* vprof = &profiles[idx*profileLen*2+profileLen];

	if (idx >= list.count)
		return;
	
	float2* profile, *reverseProf, *result;
	profile = &d_workspace[ idx * profileLength* 3 ];
	reverseProf = profile + profileLength;
	result = profile + profileLength;

	float2 pos = d_initial[idx];

	int iterations=1;
	bool boundaryHit = false;

	for (int k=0;k<1;k++) {
		float xmin = pos.x - profileLength/2;
		float ymin = pos.y - profileLength/2;

		if (images.boundaryHit(pos, profileLength/2)) {
			boundaryHit = true;
			break;
		}

		// generate X position xcor array (summing over y range)
		for (int x=0;x<profileLength;x++) {
			float s = 0.0f;
			for (int y=0;y<profileWidth;y++) {
				float xp = x * xmin;
				float yp = pos.y + (y - profileWidth/2);

				s += images.readFromTexture(xp, yp, idx);
				//s += images.interpolate(xp, yp, idx);
			}
			profile [x].x = s;
			profile [x].y = 0.0f;
			reverseProf[profileLength-x-1] = profile[x];

			printf("x profile[%d] = %f\n", x, s);
		}

	//	float offsetX = XCor1D_ComputeOffset(profile, reverseProf, result, fwkp, bwkp, profileLength);

		// generate Y position xcor array (summing over x range)
		for (int y=0;y<profileLength;y++) {
			float s = 0.0f; 
			for (int x=0;x<profileWidth;x++) {
				float xp = pos.x + (x - profileWidth/2);
				float yp = y + ymin;
				s += images.interpolate(xp, yp, idx);
			}
			profile[y].x = s;
			profile[y].y = 0.0f;
			reverseProf[profileLength-y-1] = profile[y];
		}
	}
}

void XCor1D_BuildProfiles(cudaImageListf& images, float2* d_centers, float* d_profiles, int xcorProfileLen)
{
	texture<float, cudaTextureType2D, cudaReadModeElementType> tex (0, cudaFilterModeLinear, cudaAddressModeClamp);
	
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, &tex, images.data, &desc, images.w, images.h, images.pitch);

	XCor1D_BuildProfiles_Kernel<float> <<< dim3(images.count), dim3(1) >>> (d_centers, d_profiles, xcorProfileLen, texref);
	cudaThreadSynchronize();
	cudaUnbindTexture(&tex);

}


void TestLocalizationSpeed()
{
	int repeat = 1;
	int xcorProfileLen = 32, xcorProfileWidth = 8;
	float t_gen=0, t_com=0, t_xcor=0;

	QTrkSettings cfg;
	QueuedCUDATracker qtrk(&cfg);

	cudaEvent_t gen_start, gen_end, com_start, com_end, xcor_end, xc1_profiles_end;
	cudaEventCreate(&gen_start);
	cudaEventCreate(&gen_end);
	cudaEventCreate(&com_start);
	cudaEventCreate(&com_end);
	cudaEventCreate(&xcor_end);
	cudaEventCreate(&xc1_profiles_end);

	// Create some space for images
	cudaImageListf images = cudaImageListf::alloc(170,150,1);
	dbgprintf("Image memory used: %d bytes\n", images.totalsize());
	float3* d_pos;
	cudaMalloc(&d_pos, sizeof(float3)*images.count);
	float2* d_com;
	cudaMalloc(&d_com, sizeof(float2)*images.count);
	float2* d_xcor;
	cudaMalloc(&d_xcor, sizeof(float2)*images.count);
	std::vector<float2> xcor(images.count);
	std::fill(xcor.begin(),xcor.end(), make_float2(0,0));
	cudaMemcpy(d_xcor,&xcor[0],sizeof(float2)*images.count,cudaMemcpyHostToDevice);

	float* d_profiles;
	cudaMalloc(&d_profiles, sizeof(float)*images.count*2*cfg.xc1_profileLength);

	float3* positions = new float3[images.count];
	for(int i=0;i<images.count;i++) {
		float xp = images.w/2+(rand_uniform<float>() - 0.5) * 5;
		float yp = images.h/2+(rand_uniform<float>() - 0.5) * 5;
		positions[i] = make_float3(xp, yp, 10);
	}
	cudaMemcpy(d_pos, positions, sizeof(float3)*images.count, cudaMemcpyHostToDevice);
	double comErr=0.0, xcorErr=0.0;

	for (int k=0;k<repeat;k++) {
		cudaEventRecord(gen_start);
		qtrk.GenerateImages(images, d_pos);
		cudaEventRecord(gen_end);

		cudaEventRecord(com_start);
		qtrk.ComputeBgCorrectedCOM(images, d_com);
		cudaEventRecord(com_end);
		cudaEventSynchronize(com_end);

		float t_gen0, t_com0, t_xcor0;
		cudaEventElapsedTime(&t_gen0, gen_start, gen_end);
		t_gen+=t_gen0;
		cudaEventElapsedTime(&t_com0, com_start, com_end);
		t_com+=t_com0;
		std::vector<float2> com(images.count);
		cudaMemcpyAsync(&com[0], d_com, sizeof(float2)*images.count, cudaMemcpyDeviceToHost);

		//qtrk.Compute1DXCor(images, d_com, d_xcor);
		XCor1D_BuildProfiles(images, d_com, d_profiles);

		cudaEventRecord(xcor_end);


//		qtrk.Compute1DXCorProfiles(images, d_profiles);

	//	cudaEventRecord(xc1_profiles_end);

		cudaEventSynchronize(xcor_end);
		cudaEventElapsedTime(&t_xcor0, com_end, xcor_end);
		t_xcor+=t_xcor0;

		cudaMemcpy(&xcor[0], d_xcor, sizeof(float2)*images.count, cudaMemcpyDeviceToHost);

		float dx,dy;
		for (int i=0;i<images.count;i++) {
			dx = (com[i].x-positions[i].x);
			dy = (com[i].y-positions[i].y);
			comErr += sqrt(dx*dx+dy*dy);

			dx = (xcor[i].x-positions[i].x);
			dy = (xcor[i].y-positions[i].y);
			xcorErr += sqrt(dx*dx+dy*dy);
		}
	}


	int N = images.count*repeat*1000; // times are in ms
	dbgprintf("COM error: %f pixels. XCor error: %f pixels\n",comErr/(images.count*repeat), xcorErr/(images.count*repeat));
	dbgprintf("Image generating: %f img/s. COM computation: %f img/s. 1D XCor: %f img/s\n", N/t_gen, N/t_com, N/t_xcor);
	cudaFree(d_com);
	cudaFree(d_pos);
	cudaFree(d_profiles);
	images.free();

	cudaEventDestroy(gen_start); cudaEventDestroy(gen_end); 
	cudaEventDestroy(com_start); cudaEventDestroy(com_end); 
	cudaEventDestroy(xc1_profiles_end);
}



__global__ void runCudaFFT(cudafft<float>::cpx_type *src, cudafft<float>::cpx_type *dst, cudafft<float>::KernelParams kparams)
{
	kparams.makeShared();
	cudafft<float>::transform(src,dst, kparams);
}

void TestKernelFFT()
{
	int N=256;
	cudafft<float> fft(N, false);

	std::vector< cudafft<float>::cpx_type > data(N), result(N);
	for (int x=0;x<N;x++)
		data[x].x = 10*cos(x*0.1f-5);

	fft.host_transform(&data[0], &result[0]);
	for (int x=0;x<N;x++)
		dbgprintf("[%d] %f+%fi\n", x, result[x].x, result[x].y);

	// now put data in video mem
	cudafft<float>::cpx_type *src,*d_result;
	int memSize = sizeof(cudafft<float>::cpx_type)*N;
	cudaMalloc(&src, memSize);
	cudaMemcpy(src, &data[0], memSize, cudaMemcpyHostToDevice);
	cudaMalloc(&d_result, memSize);

	int sharedMemSize = fft.kparams_size;
	for (int k=0;k<100;k++) {
		runCudaFFT<<<dim3(1),dim3(1),sharedMemSize>>>(src,d_result, fft.kparams);
	}

	std::vector< cudafft<float>::cpx_type > result2(N);
	cudaMemcpy(&result2[0], d_result, memSize, cudaMemcpyDeviceToHost);

	for (int i=0;i<N;i++) {
		cudafft<float>::cpx_type d=result2[i]-result[i];
		dbgprintf("[%d] %f+%fi\n", i, d.x, d.y);
	}

	cudaFree(src);
	cudaFree(d_result);

}

int main(int argc, char *argv[])
{
//	testLinearArray();

	std::string path = getPath(argv[0]);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	TestLocalizationSpeed();
//	TestKernelFFT();

	
	return 0;
}
