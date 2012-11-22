#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "std_incl.h"

#include "QueuedCUDATracker.h"
#include "cudaImageList.h"
#include "utils.h"

QueuedTracker* CreateQueuedTracker(QTrkSettings* cfg)
{
	return new QueuedCUDATracker(cfg);
}

QueuedCUDATracker::QueuedCUDATracker(QTrkSettings *cfg)
{
	this->cfg = *cfg;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	batchSize = numThreads * prop.multiProcessorCount;

	forward_fft = new cudafft<float>(cfg->xc1_profileLength, false);
	backward_fft = new cudafft<float>(cfg->xc1_profileLength, true);

	int sharedSpacePerThread = (prop.sharedMemPerBlock-forward_fft->kparams_size*2) / numThreads;
	dbgprintf("2X FFT instance requires %d bytes. Space per thread: %d\n", forward_fft->kparams_size*2, sharedSpacePerThread);
	dbgprintf("Device: %s\n", prop.name);
	dbgprintf("Shared memory space:%d bytes. Per thread: %d\n", prop.sharedMemPerBlock, prop.sharedMemPerBlock/numThreads);
	dbgprintf("# of CUDA processors:%d\n", prop.multiProcessorCount);
	dbgprintf("warp size: %d\n", prop.warpSize);

	int xcorWorkspaceSize = (sizeof(float2)*cfg->xc1_profileLength*3) * batchSize;
	cudaMalloc(&xcor_workspace, xcorWorkspaceSize);
	dbgprintf("XCor total required global memory: %d\n", xcorWorkspaceSize);
}

QueuedCUDATracker::~QueuedCUDATracker()
{
	delete forward_fft;
	delete backward_fft;
}

void QueuedCUDATracker::Start() 
{

}



void QueuedCUDATracker::ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, uint zlutIndex)
{
}


int QueuedCUDATracker::PollFinished(LocalizationResult* results, int maxResult)
{
	return 0;
}

void QueuedCUDATracker::SetZLUT(float* data, int planes, int res, int numLUTs)
{
}

void QueuedCUDATracker::ComputeRadialProfile(float *image, int width, int height, float* dst, int profileLength, vector2f center)
{
}



int QueuedCUDATracker::GetJobCount()
{
	return 0;
}

int QueuedCUDATracker::GetResultCount()
{
	return 0;
}


void QueuedCUDATracker::GenerateTestImage(float* dst, float xp,float yp, float z, float photoncount)
{
}


__shared__ char cudaSharedMemory[];

__global__ void compute1DXcorKernel(cudaImageList images, float2* d_initial, float2* d_xcor, float2* d_workspace,  
					cudafft<float>::KernelParams fwkp, cudafft<float>::KernelParams bwkp, int profileLength, int profileWidth)
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
	float2* profile, *reverseProf, *result;

	profile = &d_workspace[ idx * profileLength* 3 ];
	reverseProf = profile + profileLength;
	result = profile + profileLength;

	float2 pos = d_initial[idx];

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
				float xp = x * xmin;
				float yp = pos.y + (y - profileWidth/2);
				s += images.interpolate(xp, yp, idx);
			}
			profile [x].x = s;
			profile [profileLength-x-1].x = s;
		}

		xcorBuffer->XCorFFTHelper(xc, xcr, &xcorBuffer->X_result[0]);
		xcor_t offsetX = ComputeMaxInterp(&xcorBuffer->X_result[0],xcorBuffer->X_result.size()) - (xcor_t)xcorw/2;

		// generate Y position xcor array (summing over x range)
		xc = &xcorBuffer->Y_xc[0];
		xcr = &xcorBuffer->Y_xcr[0];
		for (int y=0;y<xcorw;y++) {
			xcor_t s = 0.0f; 
			for (int x=0;x<profileWidth;x++) {
				float xp = pos.x + XCorScale * (x - profileWidth/2);
				float yp = y * XCorScale + ymin;
				s += Interpolate(srcImage,width,height, xp, yp);
				MARKPIXELI(xp,yp);
			}
			xc[y] = s;
			xcr [xcorw-y-1] = xc[y];
		}

		xcorBuffer->XCorFFTHelper(xc,xcr, &xcorBuffer->Y_result[0]);
		xcor_t offsetY = ComputeMaxInterp(&xcorBuffer->Y_result[0], xcorBuffer->Y_result.size()) - (xcor_t)xcorw/2;

		pos.x += (offsetX - 1) * XCorScale * 0.5f;
		pos.y += (offsetY - 1) * XCorScale * 0.5f;
	}
	//cudafft<float>::transform(profile, 
	/*
	profile = (float2*) malloc(sizeof(float2) * profileLength * 3);
	reverseProf = profile + profileLength;
	result = profile + 2 * profileLength;
	free(profile);*/
}

void QueuedCUDATracker::Compute1DXCor(cudaImageList& images, float2* d_initial, float2* d_result)
{
	int sharedMemSize = forward_fft->kparams_size+backward_fft->kparams_size;
	compute1DXcorKernel<<<blocks(images.count), threads(), sharedMemSize >>>
		(images, d_initial, d_result, xcor_workspace, forward_fft->kparams, backward_fft->kparams, cfg.xc1_profileLength, cfg.xc1_profileWidth);
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

void QueuedCUDATracker::ComputeBgCorrectedCOM(cudaImageList& images, float2* d_com)
{
	computeBgCorrectedCOM<<<blocks(images.count), threads()>>>(images, d_com);
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


void QueuedCUDATracker::GenerateImages(cudaImageList& imgList, float3* d_pos)
{
	generateTestImages<<<blocks(imgList.count), threads()>>>(imgList, d_pos); 
}

