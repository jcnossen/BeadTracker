#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "std_incl.h"

#include "QueuedCUDATracker.h"
#include "cudaImageList.h"
#include "utils.h"

#define LSQFIT_FUNC __device__ __host__
#include "LsqQuadraticFit.h"


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

	useCPU = cfg->numThreads == 0;
}

QueuedCUDATracker::~QueuedCUDATracker()
{
	delete forward_fft;
	delete backward_fft;

	DeleteAllElems(freeBatches);
	DeleteAllElems(active);
	DeleteAllElems(jobs);
}

void QueuedCUDATracker::Start() 
{

}


void QueuedCUDATracker::ClearResults()
{
}

void QueuedCUDATracker::ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane)
{
	Batch* batch;

	if (freeBatches.empty()) { // allocate more batches?
		batch = new Batch();
	} else {
		batch = freeBatches.back();
		freeBatches.pop_back();
	}

	Job job;
	if (initialPos) job.initialPos = *initialPos;
	job.id = id;
	job.zlut = zlutIndex;
	job.locType = locType;
	batch->jobs.push_back (job);

	// Copy the image to video memory

}


int QueuedCUDATracker::PollFinished(LocalizationResult* results, int maxResult)
{
	return 0;
}

// data can be zero to allocate ZLUT data
void QueuedCUDATracker::SetZLUT(float* data,  int numLUTs, int planes, int res) 
{
}

// delete[] memory afterwards
float* QueuedCUDATracker::GetZLUT(int* planes, int *res, int *count)
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

__device__ float2 mul_conjugate(float2 a, float2 b)
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



template<typename T>
__global__ void computeBgCorrectedCOM(cudaImageList<T> images, float2* d_com)
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

void QueuedCUDATracker::ComputeBgCorrectedCOM(cudaImageListf& images, float2* d_com)
{
	computeBgCorrectedCOM<<<blocks(images.count), threads()>>>(images, d_com);
}


CUBOTH void MakeTestImage(int idx, cudaImageListf& images, float3* d_positions)
{
	float3 pos = d_positions[idx];
	
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

/*
#define KERNEL_DISPATCH(Funcname, Paramdef, Args) \
__global__ void Funcname##Kernel Paramdef { \
	int idx = blockIdx.x * blockDim.x + threadIdx.x; \
	if (idx < images.count) { \
		Funcname Args; \
	} \
} \
static void CallKernel##Funcname Paramdef { \

}

KERNEL_DISPATCH(MakeTestImage, (cudaImageListf images, float3 *d_positions), (idx, images, d_positions));
*/

void QueuedCUDATracker::GenerateImages(cudaImageListf& imgList, float3* d_pos)
{
	generateTestImages<<<blocks(imgList.count), threads()>>>(imgList, d_pos);
}

QueuedCUDATracker::Batch::~Batch() 
{
	if(imageBuf.data) imageBuf.free();
	cudaFree(d_profiles);
	delete[] hostImageBuf;
}

QueuedCUDATracker::Batch* QueuedCUDATracker::AllocBatch()
{
	Batch* b = new Batch();

	cudaMalloc(&b->d_profiles, sizeof(float)*cfg.xc1_profileLength*2*batchSize);
	b->hostImageBuf = new float[cfg.width*cfg.height*batchSize];
	b->imageBuf = cudaImageListf::alloc(cfg.width,cfg.height,batchSize);
	return b;
}

