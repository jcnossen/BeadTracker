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




void QueuedCUDATracker::ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, vector3f* initialPos, uint zlutIndex)
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


__device__ float XCor1D_ComputeOffset(float2* profile, float2* reverseProfile, float2* result, 
			cudafft<float>::KernelParams fwkp, cudafft<float>::KernelParams bwkp, int len)
{
	cudafft<float>::transform((cudafft<float>::cpx_type*) profile, (cudafft<float>::cpx_type*)result, fwkp);
	// data in 'profile' is no longer needed since we have the fourier domain version
	cudafft<float>::transform((cudafft<float>::cpx_type*) reverseProfile, (cudafft<float>::cpx_type*)profile, fwkp);

	// multiply with complex conjugate
	for (int k=0;k<len;k++)
		profile[k] = mul_conjugate(profile[k], result[k]);

	cudafft<float>::transform((cudafft<float>::cpx_type*) profile, (cudafft<float>::cpx_type*) result, bwkp);

	// shift by len/2, so the maximum will be somewhere in the middle of the array
	float* shifted = (float*)profile; // use as temp space
	for (int k=0;k<len;k++) {
		shifted[(k+len/2)%len] = result[k].x;
		printf("result[%d]=%f\n", k,result[k].x);
	}
	
	// find the interpolated maximum peak
	float maxPos = ComputeMaxInterp<float, 5>(shifted, len) - len/2;
	return maxPos;
}

//__global__ void Compute1DXCorOffsets(float

__global__ void Compute1DXcorKernel(cudaImageListf images, float2* d_initial, float2* d_xcor, float2* d_workspace,  
					cudafft<float>::KernelParams fwkp, cudafft<float>::KernelParams bwkp, int profileLength, int profileWidth)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= images.count)
		return;

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
				float xp = x + xmin;
				float yp = pos.y + (y - profileWidth/2);
				s += images.interpolate(xp, yp, idx);
			}
			profile [x].x = s;
			profile [x].y = 0.0f;
			reverseProf[profileLength-x-1] = profile[x];

			printf("x profile[%d] = %f\n", x, s);
		}

		float offsetX = XCor1D_ComputeOffset(profile, reverseProf, result, fwkp, bwkp, profileLength);

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

		float offsetY = XCor1D_ComputeOffset(profile, reverseProf, result, fwkp, bwkp, profileLength);
		pos.x += (offsetX - 1) * 0.5f;
		pos.y += (offsetY - 1) * 0.5f;
	}

	d_xcor[idx] = pos;

	//d_xcor[idx].x = 1.0f;
	//d_xcor[idx].y = 2.0f;
}




void QueuedCUDATracker::Compute1DXCor(cudaImageListf& images, float2* d_initial, float2* d_result)
{
	int sharedMemSize = forward_fft->kparams_size+backward_fft->kparams_size;
	Compute1DXcorKernel<<<blocks(images.count), threads(), sharedMemSize >>>
		(images, d_initial, d_result, xcor_workspace, forward_fft->kparams, backward_fft->kparams, cfg.xc1_profileLength, cfg.xc1_profileWidth);

/*	texture<float, cudaTextureType2D, cudaReadModeElementType> tex;
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, &tex, list->data, &desc, list->w, list->h, list->pitch);

	XCor1D_BuildProfiles_Kernel<float> <<< blocks(images.count), threads(), sharedMemSize >>> (images, 

	cudaUnbindTexture(&tex);
	*/
}


void QueuedCUDATracker::Compute1DXCorProfiles(cudaImageListf&  images, float* d_profiles)
{
	
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


__global__ void generateTestImages(cudaImageListf images, float3 *d_positions)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < images.count) {
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
}	


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

