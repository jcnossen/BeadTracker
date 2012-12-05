
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

#define LSQFIT_FUNC __device__ __host__
#include "LsqQuadraticFit.h"

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

template<typename T>
__global__ void XCor1D_BuildProfiles_Kernel(cudaImageList<T> images, float2* d_initial, T* d_profiles, int profileLength, int profileWidth)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	T* hprof = &d_profiles[idx*profileLength*2];
	T* vprof = hprof+profileLength;

	if (idx >= images.count)
		return;
	
	float2* profile, *reverseProf, *result;
	profile = (float2*)malloc(sizeof(float2)*profileLength*3);
	reverseProf = profile + profileLength;
	result = profile + profileLength*2;

	float2 pos = d_initial[idx];
//	float2 pos = make_float2(images.w/2, images.h/2);

	int iterations=1;
	bool boundaryHit = false;

	for (int k=0;k<iterations;k++) {
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
				float xp = x + xmin;
				float yp = pos.y + (y - profileWidth/2);

				s += tex2D(xcor1D_images, xp, yp + images.h * idx);
			}
			profile [x].x = s;
			profile [x].y = 0.0f;
			reverseProf[profileLength-x-1] = profile[x];
			hprof[x] = s;

			printf("x profile[%d] = %f\n", x, s);
		}
		
		//float offsetX = XCor1D_ComputeOffset(profile, reverseProf, result, fwkp, bwkp, profileLength);

		// generate Y position xcor array (summing over x range)
		for (int y=0;y<profileLength;y++) {
			float s = 0.0f; 
			for (int x=0;x<profileWidth;x++) {
				float xp = pos.x + (x - profileWidth/2);
				float yp = y + ymin;
				s += tex2D(xcor1D_images, xp, yp + images.h * idx);
			}
			profile[y].x = s;
			profile[y].y = 0.0f;
			reverseProf[profileLength-y-1] = profile[y];

			vprof[y] = s;
		}

		//float offsetY = XCor1D_ComputeOffset(profile, reverseProf, result, fwkp, bwkp, profileLength);
	}

	free(profile);
}

void XCor1D_BuildProfiles(cudaImageListf& images, float2* d_centers, float* d_profiles, int xcorProfileLen, int xcorProfileWidth)
{
	images.bind(xcor1D_images);
	XCor1D_BuildProfiles_Kernel<float> <<<dim3(1), dim3(images.count) >>> (images, d_centers, d_profiles, xcorProfileLen, xcorProfileWidth);
	cudaThreadSynchronize();
	images.unbind(xcor1D_images);

	int profileSpace=xcorProfileLen*2*images.count;
	float* profs = new float[profileSpace];
	cudaMemcpy(profs, d_profiles, sizeof(float)*profileSpace, cudaMemcpyDeviceToHost);

	WriteImageAsCSV("prof.txt", profs, xcorProfileLen*2, images.count);
	delete[] profs;
}



inline __device__ float2 mul_conjugate(float2 a, float2 b)
{
	float2 r;
	r.x = a.x*b.x + a.y*b.y;
	r.y = a.y*b.x - a.x*b.y;
	return r;
}

template<typename T>
__device__ T max_(T a, T b) { return a>b ? a : b; }
template<typename T>
__device__ T min_(T a, T b) { return a<b ? a : b; }

template<typename T, int numPts>
__device__ T ComputeMaxInterp(T* data, int len)
{
	int iMax=0;
	T vMax=data[0];
	for (int k=1;k<len;k++) {
		if (data[k]>vMax) {
			vMax = data[k];
			iMax = k;
		}
	}
	T xs[numPts]; 
	int startPos = max_(iMax-numPts/2, 0);
	int endPos = min_(iMax+(numPts-numPts/2), len);
	int numpoints = endPos - startPos;


	if (numpoints<3) 
		return iMax;
	else {
		for(int i=startPos;i<endPos;i++)
			xs[i-startPos] = i-iMax;

		LsqSqQuadFit<T> qfit(numpoints, xs, &data[startPos]);
		//printf("iMax: %d. qfit: data[%d]=%f\n", iMax, startPos, data[startPos]);
		//for (int k=0;k<numpoints;k++) {
	//		printf("data[%d]=%f\n", startPos+k, data[startPos]);
		//}
		T interpMax = qfit.maxPos();

		if (fabs(qfit.a)<1e-9f)
			return (T)iMax;
		else
			return (T)iMax + interpMax;
	}
}

__device__ float XCor1D_ProfileFFT(float2* profile, float2* reverseProfile, float2* result, 
			cudafft<float>::KernelParams fwkp, cudafft<float>::KernelParams bwkp, int len, float *dbgout)
{
	cudafft<float>::transform((cudafft<float>::cpx_type*) profile, (cudafft<float>::cpx_type*)result, fwkp);
	// data in 'profile' is no longer needed since we have the fourier domain version
	cudafft<float>::transform((cudafft<float>::cpx_type*) reverseProfile, (cudafft<float>::cpx_type*)profile, fwkp);

	// multiply with complex conjugate
	for (int k=0;k<len;k++)
		profile[k] = mul_conjugate(profile[k], result[k]);

	cudafft<float>::transform((cudafft<float>::cpx_type*) profile, (cudafft<float>::cpx_type*) result, bwkp);

	// shift by len/2, so the maximum will be somewhere in the middle of the array
	float* shifted = dbgout;
	for (int k=0;k<len;k++) {
		shifted[(k+len/2)%len] = result[k].x;
		//printf("result[%d]=%f\n", k,result[k].x);
	}
	
	// find the interpolated maximum peak
	float maxPos = ComputeMaxInterp<float, 5>(shifted, len) - len/2;
	return maxPos;
}



__global__ void XCor1D_ComputeOffsets(float* d_profiles, float2* d_positions, int profileLen, cudafft<float>::KernelParams fwkp, cudafft<float>::KernelParams bwkp)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float2* profile, *reverseProf, *result;
	profile = (float2*)malloc(sizeof(float2)*profileLen*3);
	reverseProf = profile + profileLen;
	result = profile + profileLen*2;

	float* hprof = &d_profiles[idx*profileLen*2];
	float* vprof = hprof+profileLen;

	for (int k=0;k<profileLen;k++)
		profile[k] = make_float2(hprof[k],0.0f);
	float offsetX = XCor1D_ProfileFFT(profile, reverseProf, result, fwkp, bwkp, profileLen, hprof);

	for (int k=0;k<profileLen;k++)
		profile[k] = make_float2(vprof[k],0.0f);
	float offsetY = XCor1D_ProfileFFT(profile, reverseProf, result, fwkp, bwkp, profileLen, vprof);
	

	printf("[%d] offsetX: %f, offsetY: %f\n", idx, offsetX,offsetY);

	d_positions[idx].x += (offsetX - 1)*0.5f;
	d_positions[idx].y += (offsetY - 1)*0.5f;

	free(profile);
}

void ComputeOffsets(cudaImageListf& images, float* d_profiles, float2* d_positions, int profileLen, cudafft<float>::KernelParams fwkp, cudafft<float>::KernelParams bwkp)
{
	XCor1D_ComputeOffsets<<<dim3(1), dim3(images.count)>>> (d_profiles, d_positions, profileLen,fwkp, bwkp);

	int profileSpace=profileLen*2*images.count;
	float* profs = new float[profileSpace];
	cudaMemcpy(profs, d_profiles, sizeof(float)*profileSpace, cudaMemcpyDeviceToHost);

	WriteImageAsCSV("fdprof.txt", profs, profileLen*2, images.count);
	delete[] profs;
}

//void XCor1D

texture<float, cudaTextureType2D, cudaReadModeElementType> smpImgRef(0, cudaFilterModeLinear);

__global__ void smpImageKernel(cudaImageListf images, float2 center, int w, int h, float* dst, float zoom)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= images.count)
		return;
	
	float halfX = w*0.5f;
	float halfY = h*0.5f;
	for (int y=0;y<h;y++)
		for (int x=0;x<w;x++){
			 dst[y*w+x] = tex2D( smpImgRef, center.x + (x - halfX) * zoom, center.y + ( y - halfY ) * zoom + idx*h );
		}
}

void SampleImage(cudaImageListf& images, float2 center, int w, int h)
{
	float* dst;
	cudaMalloc(&dst,sizeof(float)*w*h);
	images.bind(smpImgRef);
	smpImageKernel<<<dim3(1),dim3(images.count)>>> (images,center,w,h,dst, 0.5f);
	float* tmp = (float*)ALLOCA(sizeof(float)*w*h);
	cudaMemcpy(tmp, dst, sizeof(float)*w*h, cudaMemcpyDeviceToHost);
	cudaFree(dst);
	images.unbind(smpImgRef);

	WriteImageAsCSV("imgsmp.txt", tmp, w,h);
}


void TestLocalizationSpeed()
{
	int repeat = 1;
	int numImages=60;
	int xcorProfileLen = 128, xcorProfileWidth = 16;
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
	cudaImageListf images = cudaImageListf::alloc(170,150,numImages);
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
	cudaMalloc(&d_profiles, sizeof(float)*images.count*3*cfg.xc1_profileLength);

	float3* positions = new float3[images.count];
	for(int i=0;i<images.count;i++) {
		float xp = images.w/2+(rand_uniform<float>() - 0.5) * 5;
		float yp = images.h/2+(rand_uniform<float>() - 0.5) * 5;
		positions[i] = make_float3(xp, yp, 3);
		dbgprintf("Pos[%d]=( %f, %f )\n", i, xp, yp);
	}
	cudaMemcpy(d_pos, positions, sizeof(float3)*images.count, cudaMemcpyHostToDevice);
	double comErr=0.0, xcorErrX=0.0, xcorErrY=0.0;

	cudafft<float> fwfft(xcorProfileLen, true) ,bwfft(xcorProfileLen,false);

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
		cudaMemcpyAsync(d_xcor, d_com, sizeof(float2)*images.count, cudaMemcpyDeviceToDevice);

		//qtrk.Compute1DXCor(images, d_com, d_xcor);

		SampleImage(images, com[0], 30,30);

		XCor1D_BuildProfiles(images, d_com, d_profiles, xcorProfileLen, xcorProfileWidth);

		ComputeOffsets(images, d_profiles, d_xcor, xcorProfileLen, fwfft.kparams, bwfft.kparams);
		cudaEventRecord(xcor_end);
		cudaEventSynchronize(xcor_end);

		cudaEventElapsedTime(&t_xcor0, com_end, xcor_end);
		t_xcor+=t_xcor0;

		cudaMemcpy(&xcor[0], d_xcor, sizeof(float2)*images.count, cudaMemcpyDeviceToHost);

		float dx,dy;
		for (int i=0;i<images.count;i++) {
			dx = (com[i].x-positions[i].x);
			dy = (com[i].y-positions[i].y);
			comErr += sqrt(dx*dx+dy*dy);

			dx = (xcor[i].x-positions[i].x)+1.5f;
			dy = (xcor[i].y-positions[i].y)+1.5f;
			xcorErrX += dx;
			xcorErrY += dy;
		}
	}


	int N = images.count*repeat*1000; // times are in ms
	dbgprintf("COM error: %f pixels. XCor error: [X %f, Y %f] pixels\n",comErr/(images.count*repeat), xcorErrX/(images.count*repeat),xcorErrY/(images.count*repeat));
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
