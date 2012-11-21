
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


double getPreciseTime()
{
	uint64_t freq, time;

	QueryPerformanceCounter((LARGE_INTEGER*)&time);
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	return (double)time / (double)freq;
}


#define CB __device__ __host__



std::string getPath(const char *file)
{
	std::string s = file;
	int pos = s.length()-1;
	while (pos>0 && s[pos]!='\\' && s[pos]!= '/' )
		pos--;
	
	return s.substr(0, pos);
}


struct cudaImageList {
	// No constructor used to allow passing as CUDA kernel argument
	float* data;
	size_t pitch;
	int w,h;
	int count;
	enum { numThreads=64 };
	dim3 blocks() {
		return dim3((count+numThreads-1)/numThreads);
	}
	dim3 threads() {
		return dim3(numThreads);
	}

	static cudaImageList alloc(int w,int h, int amount) {
		cudaImageList imgl;
		imgl.w = w; imgl.h = h;
		imgl.count = amount;
		cudaMallocPitch(&imgl.data, &imgl.pitch, sizeof(float)*w, h*amount);
		return imgl;
	}

	CB float* get(int i) {
		return (float*)(((char*)data) + pitch*h*i);
	}

	CB float& pixel(int x,int y, int imgIndex) {
		float* row = (float*) ( (char*)data + (h*imgIndex+y)*pitch );
		return row[x];
	}

	void free()
	{
		cudaFree(data);
	}

	int totalsize() { return pitch*h*count; }
};


__shared__ char cudaSharedMemory[];

__global__ void compute1DXcorKernel(cudaImageList images, float2* d_initial, float2* d_xcor, cudafft<float>::KernelParams fwkp, cudafft<float>::KernelParams bwkp)
{
	char* fft_fw_shared = cudaSharedMemory;
	char* fft_bw_shared = cudaSharedMemory + fwkp.memsize;
	char* fftdata[] = { fwkp.data, bwkp.data };
	if (threadIdx.x < 2) { // thread 0 copies forward FFT data, thread 1 copies backward FFT data. Thread 2-31 can relax
		memcpy(cudaSharedMemory + threadIdx.x * fwkp.memsize, fftdata[threadIdx.x], fwkp.memsize);
	}

	// Now we can forget about the global memory ptrs, as the FFT parameter data is now stored in shared memory
	fwkp.data = fft_fw_shared;
	bwkp.data = fft_bw_shared;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
}

void compute1DXcor(cudaImageList& images, float2* d_initial, float2* d_xcor, cudafft<float>& forward_fft, cudafft<float>& backward_fft)
{
	int sharedMemSize = forward_fft.kparams_size+backward_fft.kparams_size;
	compute1DXcorKernel<<<images.blocks(), images.threads(), sharedMemSize >>>(images, d_initial, d_xcor, forward_fft.kparams, backward_fft.kparams);
}


__global__ void computeBgCorrectedCOM(cudaImageList images, float2* d_com)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int imgsize = images.w*images.h;
	float sum=0, sum2=0;
	float momentX=0;
	float momentY=0;

	if (idx < images.count) {

		for (int y=0;y<images.h;y++)
			for (int x=0;x<images.w;x++) {
				float v = images.pixel(x,y,idx);
				sum += v;
				sum2 += v*v;
			}

		float invN = 1.0f/imgsize;
		float mean = sum * invN;
		float stdev = sqrtf(sum2 * invN - mean * mean);
		sum = 0.0f;

		for (int y=0;y<images.h;y++)
			for(int x=0;x<images.w;x++)
			{
				float v = images.pixel(x,y,idx);
				v = fabs(v-mean)-2.0f*stdev;
				if(v<0.0f) v=0.0f;
				sum += v;
				momentX += x*v;
				momentY += y*v;
			}

		d_com[idx].x = momentX / (float)sum;
		d_com[idx].y = momentY / (float)sum;
	}
}


__global__ void generateTestImages(cudaImageList images, float3 *d_positions)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float3 pos = d_positions[idx];
	
	if (idx < images.count) {
		float S = 1.0f/pos.z;
		for (int y=0;y<images.h;y++) {
			for (int x=0;x<images.w;x++) {
				float X = x - pos.x;
				float Y = y - pos.y;
				float r = sqrtf(X*X+Y*Y)+1;
				float v = sinf(r/(5*S)) * expf(-r*r*S*0.01f);
				images.pixel(x,y,idx) = v;
			}
		}
	}
}	

int main(int argc, char *argv[])
{
//	testLinearArray();
	int repeat = 10;
	int xcorw = 128;

	std::string path = getPath(argv[0]);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	float t_gen=0, t_com=0, t_xcor=0;

	cudaEvent_t gen_start, gen_end, com_start, com_end, xcor_end;
	cudaEventCreate(&gen_start);
	cudaEventCreate(&gen_end);
	cudaEventCreate(&com_start);
	cudaEventCreate(&com_end);
	cudaEventCreate(&xcor_end);

	dbgprintf("Device: %s\n", prop.name);
	dbgprintf("Shared memory space:%d bytes\n", prop.sharedMemPerBlock);
	dbgprintf("# of CUDA processors:%d\n", prop.multiProcessorCount);
	dbgprintf("warp size: %d\n", prop.warpSize);

	// Create some space for images
	cudaImageList images = cudaImageList::alloc(170,150, 2048);
	dbgprintf("Image memory used: %d bytes\n", images.totalsize());
	float3* d_pos;
	cudaMalloc(&d_pos, sizeof(float3)*images.count);
	float2* d_com;
	cudaMalloc(&d_com, sizeof(float2)*images.count);
	float2* d_xcor;
	cudaMalloc(&d_xcor, sizeof(float2)*images.count);

	float3* positions = new float3[images.count];
	for(int i=0;i<images.count;i++) {
		float xp = images.w/2+(rand_uniform<float>() - 0.5) * 5;
		float yp = images.h/2+(rand_uniform<float>() - 0.5) * 5;
		positions[i] = make_float3(xp, yp, 10);
	}
	cudaMemcpy(d_pos, positions, sizeof(float3)*images.count, cudaMemcpyHostToDevice);
	double err=0.0;
	cudafft<float> forward_fft(xcorw, false);
	cudafft<float> backward_fft(xcorw, true);
	dbgprintf("FFT instance requires %d bytes\n", forward_fft.kparams_size);

	for (int k=0;k<repeat;k++) {
		cudaEventRecord(gen_start);
		generateTestImages<<<images.blocks(), images.threads()>>>(images, d_pos); 
		cudaEventRecord(gen_end);

		cudaEventRecord(com_start);
		computeBgCorrectedCOM<<<images.blocks(), images.threads()>>>(images, d_com);
		cudaEventRecord(com_end);
		cudaEventSynchronize(com_end);

		float t_gen0, t_com0, t_xcor0;
		cudaEventElapsedTime(&t_gen0, gen_start, gen_end);
		t_gen+=t_gen0;
		cudaEventElapsedTime(&t_com0, com_start, com_end);
		t_com+=t_com0;
		std::vector<float2> com(images.count);
		cudaMemcpyAsync(&com[0], d_com, sizeof(float2)*images.count, cudaMemcpyDeviceToHost);
		compute1DXcor(images, d_com, d_xcor, forward_fft, backward_fft);
		cudaEventRecord(xcor_end);
		cudaEventSynchronize(xcor_end);
		cudaEventElapsedTime(&t_xcor0, com_end, xcor_end);
		t_xcor+=t_xcor0;

		for (int i=0;i<images.count;i++) {
			float dx = (com[i].x-positions[i].x);
			float dy = (com[i].y-positions[i].y);
			err += sqrt(dx*dx+dy*dy);
		}
	}


	int N = images.count*repeat*1000; // times are in ms
	dbgprintf("COM error: %f pixels\n", err/(images.count*repeat));
	dbgprintf("Image generating: %f img/s. COM computation: %f img/s.\n", N/t_gen, N/t_com);
	cudaFree(d_com);
	cudaFree(d_pos);
	images.free();

	cudaEventDestroy(gen_start); cudaEventDestroy(gen_end); 
	cudaEventDestroy(com_start); cudaEventDestroy(com_end); 
	
	return 0;
}
