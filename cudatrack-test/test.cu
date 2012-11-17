
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "std_incl.h"
#include "utils.h"

#include <cassert>
#include <cstdlib>
#include <stdio.h>
#include <windows.h>
#include <cstdarg>

#include "cuda_kissfft/kiss_fft.h"
#include "random_distr.h"

static DWORD startTime = 0;
void BeginMeasure() { startTime = GetTickCount(); }
DWORD EndMeasure(const std::string& msg) {
	DWORD endTime = GetTickCount();
	DWORD dt = endTime-startTime;
	dbgprintf("%s: %d ms\n", msg.c_str(), dt);
	return dt;
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


__global__ void computeBgCorrectedCOM(float* d_images, int width,int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int imgsize = width*height;
	float* data = &d_images[imgsize*idx];
	float sum=0, sum2=0;
	float momentX=0;
	float momentY=0;

	for (int y=0;y<height;y++)
		for (int x=0;x<width;x++) {
			float v = data[y*width+x];
			sum += v;
			sum2 += v*v;
		}

	float invN = 1.0f/(width*height);
	float mean = sum * invN;
	float stdev = sqrtf(sum2 * invN - mean * mean);
	sum = 0.0f;

	for (int y=0;y<height;y++)
		for(int x=0;x<width;x++)
		{
			float v = data[y*width+x];
			v = fabs(v-mean)-2.0f*stdev;
			if(v<0.0f) v=0.0f;
			sum += v;
			momentX += x*v;
			momentY += y*v;
		}
	vector2f com;
	com.x = momentX / (float)sum;
	com.y = momentY / (float)sum;
}


struct cudaImageList {
	// No constructor used to allow passing as CUDA kernel argument
	float* data;
	size_t pitch;
	int w,h;
	int count;

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

	void free()
	{
		cudaFree(data);
	}
};


__global__ void generateTestImages(cudaImageList images, float3 *d_positions)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	float3 pos = d_positions[idx];
	float* data = images.get(idx);
	
	float S = 1.0f/pos.z;
	for (int y=0;y<images.h;y++) {
		for (int x=0;x<images.w;x++) {
			float X = x - pos.x;
			float Y = y - pos.y;
			float r = sqrtf(X*X+Y*Y)+1;
			float v = sinf(r/(5*S)) * expf(-r*r*S*0.01f);
			data[y*images.pitch+x] = v;
		}
	}
}	

int main(int argc, char *argv[])
{
//	testLinearArray();

	std::string path = getPath(argv[0]);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	dbgprintf("Shared memory space:%d bytes\n", prop.sharedMemPerBlock);

	// Create some space for images
	cudaImageList images = cudaImageList::alloc(170,150, 100);

	float3* positions = new float3[images.count];
	for(int i=0;i<images.count;i++) {
		float xp = images.w/2+(rand_uniform<float>() - 0.5) * 5;
		float yp = images.h/2+(rand_uniform<float>() - 0.5) * 5;
		positions[i] = make_float3(xp, yp, 10);
	}
	float3* d_pos;
	cudaMalloc(&d_pos, sizeof(float3)*images.count);
	cudaMemcpy(d_pos, positions, sizeof(float3)*images.count, cudaMemcpyHostToDevice);
	dim3 nBlocks = dim3(1,1,1);
	generateTestImages<<<nBlocks, dim3(images.count,1,1)>>>(images, d_pos); 

	cudaFree(d_pos);
	images.free();
	
	return 0;
}
