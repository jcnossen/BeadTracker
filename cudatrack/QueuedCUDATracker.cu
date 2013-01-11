/*

Quadrant Interpolation on CUDA
This implementation is designed to perform QI on many small images, processing each image on 2 CUDA processors.

Layout of CUDA shared memory:

Per warp:
- float2[radialsteps*4]: twiddles

Per localization job (16 per warp)
- float2: Current position

Per thread( 32 per warp )
- float2[radialsteps*2]: Temporary buffer 1
- float2[radialsteps*2]: Temporary buffer 2

*/

#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "std_incl.h"
#include "vector_types.h"

#include "QueuedCUDATracker.h"
#include "gpu_utils.h"
#include "simplefft.h"

#include "utils.h"

#define LSQFIT_FUNC __device__ __host__
#include "LsqQuadraticFit.h"

// Types used by QI algorithm
typedef float qivalue_t;
typedef sfft::complex<qivalue_t> qicomplex_t;

// This template specialization makes sure that we dont link against cudaSharedMemory from host-side code (You get a linker error if you do)
__shared__ float2 cudaSharedMemory[];

// QueuedCUDATracker allows runtime choosing of GPU or CPU code. All GPU kernel calls are done through the following macro:
// Depending on 'useCPU' it either invokes a CUDA kernel named 'Funcname', or simply loops over the data on the CPU side calling 'Funcname' for each image
#define KERNEL_DISPATCH(Funcname, TParam) \
__global__ void Funcname##Kernel(cudaImageListf images, TParam param, int sharedMemPerThread) { \
	int idx = blockIdx.x * blockDim.x + threadIdx.x; \
	if (idx < images.count) { \
		Funcname(idx, images, &cudaSharedMemory [threadIdx.x * sharedMemPerThread], param); \
	} \
} \
void QueuedCUDATracker::CallKernel_##Funcname(cudaImageListf& images, TParam param, uint sharedMemPerThread)  { \
	if (useCPU) { \
		for (int idx=0;idx<images.count;idx++) { \
			::Funcname(idx, images, sharedBuf.data, param); \
		} \
	} else { \
		Funcname##Kernel <<<blocks(images.count), threads(), sharedMemPerThread * numThreads >>> (images,param, sharedMemPerThread); \
	} \
}

QueuedTracker* CreateQueuedTracker(QTrkSettings* cfg)
{
	return new QueuedCUDATracker(cfg);
}

QueuedCUDATracker::QueuedCUDATracker(QTrkSettings *cfg, int batchSize)
{
	this->cfg = *cfg;

	cudaGetDeviceProperties(&deviceProp, 0);

	if(batchSize<0) batchSize = numThreads * deviceProp.multiProcessorCount;
	this->batchSize = batchSize;
	maxActiveBatches = std::min(2, (cfg->maxQueueSize + batchSize - 1) / batchSize);

	//int sharedSpacePerThread = (prop.sharedMemPerBlock-forward_fft->kparams_size*2) / numThreads;
//	dbgprintf("2X FFT instance requires %d bytes. Space per thread: %d\n", forward_fft->kparams_size*2, sharedSpacePerThread);
	dbgprintf("Device: %s\n", deviceProp.name);
	dbgprintf("Shared memory space:%d bytes. Per thread: %d\n", deviceProp.sharedMemPerBlock, deviceProp.sharedMemPerBlock/numThreads);
	dbgprintf("# of CUDA processors:%d\n", deviceProp.multiProcessorCount);
	dbgprintf("warp size: %d\n", deviceProp.warpSize);

	useCPU = cfg->numThreads == 0;
	qiProfileLen = 1;
	while (qiProfileLen < cfg->qi_radialsteps*2) qiProfileLen *= 2;

	// See file header comment for shared memory layout
	sharedMemSize = sizeof(float2) * 2*qiProfileLen + // twiddles
		sizeof(float2) * 16+
	sizeof(float2) *qiProfileLen * 32;
		
	fft_twiddles = DeviceMem( sfft::fill_twiddles<float> (qiProfileLen) );
	KernelParams &p = kernelParams;
	
	ZLUTParams& zp = p.zlut;
	zp.angularSteps = cfg->zlut_angularsteps;
	zp.maxRadius = cfg->zlut_maxradius;
	zp.minRadius = cfg->zlut_minradius;

	QIParams& qi = p.qi;
	qi.angularSteps = cfg->qi_angularsteps;
	qi.iterations = cfg->qi_iterations;
	qi.maxRadius = cfg->qi_maxradius;
	qi.minRadius = cfg->qi_minradius;
	qi.radialSteps = cfg->qi_radialsteps;
	qi.d_twiddles = fft_twiddles.data;
	p.sharedBuf = sharedBuf.data;
	//p.useShared = qi.radialSteps*2 * numThreads < deviceProp.sharedMemPerBlock;

	dbgprintf("Required shared memory: %d\n", sharedMemSize );
	currentBatch = AllocBatch();
	
	int bufperimg = sizeof(qivalue_t)*qiProfileLen*2 + sizeof(qicomplex_t)*qiProfileLen;
	buffer.init(batchSize * bufperimg);
	kernelParams.buffer = buffer.data;

	zlut = cudaImageListf::empty();
}

QueuedCUDATracker::~QueuedCUDATracker()
{
	DeleteAllElems(freeBatches);
	DeleteAllElems(active);
}

void QueuedCUDATracker::GenerateTestImage(float* dst, float xp,float yp, float z, float photoncount)
{
}

static CUBOTH float2 BgCorrectedCOM(int idx, cudaImageListf& images)
{
	int imgsize = images.w*images.h;
	float sum=0, sum2=0;
	float momentX=0;
	float momentY=0;

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

	float2 com;
	com.x = momentX / (float)sum;
	com.y = momentY / (float)sum;
	return com;
}

static CUBOTH void BgCorrectedCOM(int idx, cudaImageListf& images, float2* sharedMem, float2* d_com) {
	d_com[idx] = BgCorrectedCOM(idx, images);
}

KERNEL_DISPATCH(BgCorrectedCOM, float2*);

void QueuedCUDATracker::ComputeBgCorrectedCOM(cudaImageListf& images, float2* d_com)
{
	CallKernel_BgCorrectedCOM(images, d_com);
}


static CUBOTH void MakeTestImage(int idx, cudaImageListf& images, float2* sharedMem, float3* d_positions)
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


KERNEL_DISPATCH(MakeTestImage, float3*); 

void QueuedCUDATracker::GenerateImages(cudaImageListf& imgList, float3* d_pos)
{
	CallKernel_MakeTestImage(imgList, d_pos);
}


texture<float, cudaTextureType2D, cudaReadModeElementType> qi_image_texture(0, cudaFilterModeLinear);

static __device__ void RadialProfile(int idx, cudaImageListf& images, float *dst, float2* tmpbuf, const ZLUTParams zlut, float2 center, bool& error)
{
	for (int i=0;i<zlut.radialSteps;i++)
		dst[i]=0.0f;

	float totalrmssum2 = 0.0f;
	float rstep = (zlut.maxRadius-zlut.minRadius) / zlut.radialSteps;
	for (int i=0;i<zlut.radialSteps; i++) {
		float sum = 0.0f;

		float r = zlut.minRadius+rstep*i;
		for (int a=0;a<zlut.angularSteps;a++) {
			float ang = 2*3.141593f*a/(float)zlut.angularSteps;
			float x = center.x + __cosf(ang) * r;
			float y = center.y + __sinf(ang) * r;
			sum += images.interpolate(x,y, idx);
		}

		dst[i] = sum/zlut.angularSteps-images.borderValue;
		totalrmssum2 += dst[i]*dst[i];
	}
	double invTotalrms = 1.0f/sqrt(totalrmssum2/zlut.radialSteps);
	for (int i=0;i<zlut.radialSteps;i++) {
		dst[i] *= invTotalrms;
	}
}


static __device__ qivalue_t QI_ComputeOffset(qicomplex_t* profile, qicomplex_t* tmpbuf1, const QIParams& params, int idx, sfft::complex<float>* s_twiddles) {
	int nr = params.radialSteps;

	qicomplex_t* reverse = tmpbuf1;

	for(int x=0;x<nr*2;x++)
		reverse[x] = profile[nr*2-1-x];

//	std::vector< sfft::complex<float> > tw = sfft::fill_twiddles<float> (nr*2);
	sfft::fft_forward(nr*2, profile, s_twiddles);
	sfft::fft_forward(nr*2, reverse, s_twiddles);

	// multiply with conjugate
	for(int x=0;x<nr*2;x++)
		profile[x] = profile[x] * reverse[x].conjugate();

	sfft::fft_inverse(nr*2, profile, s_twiddles);
	// fft_out2 now contains the autoconvolution
	// convert it to float
	qivalue_t* autoconv = (qivalue_t*)reverse;
	for(int x=0;x<nr*2;x++)  {
		autoconv[x] = profile[(x+nr)%(nr*2)].real();
	}

	float maxPos = ComputeMaxInterp<qivalue_t,7>(autoconv, nr*2);
	//free(reverse);
	return (maxPos - nr) / (3.14159265359f * 0.5f);
}


static __device__ void ComputeQuadrantProfile(cudaImageListf& images, int idx, float* dst, const QIParams& params, int quadrant, float2 center)
{
	const int qmat[] = {
		1, 1,
		-1, 1,
		-1, -1,
		1, -1 };
	int mx = qmat[2*quadrant+0];
	int my = qmat[2*quadrant+1];

	for (int i=0;i<params.radialSteps;i++)
		dst[i]=0.0f;
	
	double total = 0.0f;
	float rstep = (params.maxRadius - params.minRadius) / params.radialSteps;
	for (int i=0;i<params.radialSteps; i++) {
		double sum = 0.0f;
		float r = params.minRadius + rstep * i;

		for (int a=0;a<params.angularSteps;a++) {
			float ang = 0.5f*3.141593f*a/(float)params.angularSteps;
			float x = center.x + mx*cos(ang) * r;
			float y = center.y + my*sin(ang) * r;
			sum += images.interpolate(x, y, idx);
		}

		dst[i] = sum/params.angularSteps-images.borderValue;
		total += dst[i];
	}
}

template<typename T>
static __device__ void CopyToShared(T* dst, T* src, int numElem)
{
	int elemPerThread = (numElem + blockDim.x-1) / blockDim.x;
	int offset = elemPerThread * threadIdx.x;
	int endpos = min(numElem, offset + elemPerThread);
	for (int x=offset;x<endpos;x++)
		dst[x] = src[x];
}


static __device__ float2 ComputeQIPosition(int idx, cudaImageListf& images, KernelParams params, float2 initial, bool& error)
{
	// Prepare shared memory
	int pl = params.qi.radialSteps*2;
	float2* sharedBuf = params.sharedBuf;
	int imgThread = threadIdx.x&1; // either 0 or 1. 0 does the X coordinate, 1, does the Y coordinate
	
	// Copy twiddles to shared
	sfft::complex<float>* s_twiddles = (sfft::complex<float>*)&sharedBuf[0];
	CopyToShared(s_twiddles, params.qi.d_twiddles, pl*2); // twiddles are pl*2 (first the forward version, then the inverse version)

	sharedBuf += pl*2;
	float2* s_currentPos = &sharedBuf[threadIdx.x/2];

	int sharedElemPerImg = pl;
	sharedBuf += 16;
	float2* s_tmpbuf0 = &sharedBuf[threadIdx.x * sharedElemPerImg];
	float2* s_tmpbuf1 = &sharedBuf[threadIdx.x * sharedElemPerImg + pl];

	// Localize
	QIParams& qp = params.qi;
	int nr=qp.radialSteps;
	float2 center = initial;
	float pixelsPerProfLen = (qp.maxRadius-qp.minRadius)/qp.radialSteps;

	size_t total_required = sizeof(qivalue_t)*nr*4 + sizeof(qicomplex_t)*nr*2;

	error = false;

	qivalue_t* buf = (qivalue_t*)&params.buffer[total_required * idx];
	qivalue_t* q0=buf, *q1=buf+nr, *q2=buf+nr*2, *q3=buf+nr*3;

	qicomplex_t* concat0 = (qicomplex_t*)  (qicomplex_t*)(buf + nr*4);
	qicomplex_t* concat1 = concat0 + nr;
	for (int k=0;k<qp.iterations;k++) {
		// check bounds
		error = images.boundaryHit(center, qp.maxRadius);

		for (int q=0;q<4;q++) {
			ComputeQuadrantProfile(images, idx, buf+q*nr, qp, q, center);
		}
		
		// Build Ix = qL(-r) || qR(r)
		// qL = q1 + q2   (concat0)
		// qR = q0 + q3   (concat1)
		for(int r=0;r<nr;r++) {
			concat0[nr-r-1] = qicomplex_t(q1[r]+q2[r]);
			concat1[r] = qicomplex_t(q0[r]+q3[r]);
		}

		
		float offsetX = QI_ComputeOffset(concat0, (qicomplex_t*) s_tmpbuf0, qp, idx, s_twiddles);

		// Build Iy = qB(-r) || qT(r)
		// qT = q0 + q1
		// qB = q2 + q3
		for(int r=0;r<nr;r++) {
			concat0[r] = qicomplex_t(q0[r]+q1[r]);
			concat1[nr-r-1] = qicomplex_t(q2[r]+q3[r]);
		}
		float offsetY = QI_ComputeOffset(concat0, (qicomplex_t*)s_tmpbuf0, qp, idx, s_twiddles);

		//printf("[%d] OffsetX: %f, OffsetY: %f\n", k, offsetX, offsetY);
		center.x += offsetX * pixelsPerProfLen;
		center.y += offsetY * pixelsPerProfLen;
	}

	//free(buf);
	return center;
}



__global__ void ComputeQIKernel(cudaImageListf images, KernelParams params, float2* d_initial, float2* d_result)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < images.count) {
		bool error;
		params.sharedBuf = cudaSharedMemory;
		d_result[idx] = ComputeQIPosition(idx, images, params, d_initial[idx], error);
	}
}

void QueuedCUDATracker::ComputeQI(cudaImageListf& images, float2* d_initial, float2* d_result)
{
	images.bind(qi_image_texture);
	dim3 threads(32);

	ComputeQIKernel<<< blocks(images.count), threads, sharedMemSize >>> (images, kernelParams, d_initial, d_result);
	images.unbind(qi_image_texture);
}

QueuedCUDATracker::Batch::~Batch() 
{
	if(images.data) images.free();
	cudaFreeHost(hostImageBuf);
	cudaEventDestroy(localizationDone);
	cudaEventDestroy(imageBufferCopied);
}

QueuedCUDATracker::Batch* QueuedCUDATracker::AllocBatch()
{
	if (freeBatches.empty()) { // allocate more batches?
		Batch* b = new Batch();
		
		uint hostBufSize = sizeof(float)* cfg.width*cfg.height*batchSize;
		cudaMallocHost(&b->hostImageBuf, hostBufSize, cudaHostAllocWriteCombined);
		b->images = cudaImageListf::alloc(cfg.width,cfg.height,batchSize, useCPU);
		b->d_jobs.init(batchSize);
		cudaEventCreate(&b->localizationDone);
		cudaEventCreate(&b->imageBufferCopied);

		return b;
	} else {
		Batch* batch = freeBatches.back();
		freeBatches.pop_back();
		return batch;
	}
}



void QueuedCUDATracker::Start() 
{

}


void QueuedCUDATracker::ClearResults()
{
	FetchResults();
	results.clear();
}

__global__ void LocalizeBatchKernel(int numImages, cudaImageListf images, KernelParams params, CUDATrackerJob* jobs)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx >= numImages)
		return;

	float2 com = BgCorrectedCOM(idx, images);

	CUDATrackerJob* j = &jobs[idx];

	LocalizeType locType = (LocalizeType)(j->locType&Localize2DMask);
	bool boundaryHit = false;
	float2 result = com;
	//float* tmpbuf = &params.buffer[params.bufferMemPerThread * threadIdx.x];
	//float* sharedtmpbuf 

	if (locType == LocalizeQI) {
		bool error;
		result = ComputeQIPosition(idx, images, params, com, error);
		j->error = error?1:0;
	}

	if (j->locType & LocalizeBuildZLUT) {
		float* d_zlut = params.zlut.GetZLUT(j->zlut, j->zlutPlane);
		RadialProfile(idx, images, d_zlut, params.sharedBuf, params.zlut, result, boundaryHit);
	}
	else if (j->locType & LocalizeZ) {
		float* d_zlut = params.zlut.GetZLUT(j->zlut, 0);
		//RadialProfile(idx, images, , params.sharedBuf, params.zlut, result, boundaryHit);
	}

	j->firstGuess = com;
	j->resultPos.x = result.x;
	j->resultPos.y = result.y;
}

bool QueuedCUDATracker::IsIdle()
{
	FetchResults();
	return active.empty () && currentBatch->jobs.empty();
}


bool QueuedCUDATracker::IsQueueFilled()
{
	FetchResults();
	return active.size() >= maxActiveBatches;
}

bool QueuedCUDATracker::ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane)
{
	if (IsQueueFilled()) {
		Threads::Sleep(5);
	}

	CUDATrackerJob job;
	if (initialPos) 
		job.initialPos = *(float3*)initialPos;
	job.id = id;
	job.zlut = zlutIndex;
	job.locType = locType;

	Batch* cb = currentBatch;
	cb->jobs.push_back(job);

	// Copy the image to the batch image buffer (CPU side)
	float* hostbuf = &cb->hostImageBuf[cfg.height*cfg.width* (cb->jobs.size()-1)];
	uchar* srcptr = data;
	CopyImageToFloat(data, cfg.width, cfg.height, pitch, pdt, hostbuf);

	// If batch is filled, copy the image to video memory asynchronously, and start the localization
	if (cb->jobs.size() == batchSize)
		QueueCurrentBatch();

	return true;
}


void QueuedCUDATracker::QueueCurrentBatch()
{
	Batch* cb = currentBatch;

	if (cb->jobs.empty())
		return;

	dbgprintf("Sending %d images to GPU...\n", cb->jobs.size());

	cudaMemcpy2DAsync(cb->images.data, cb->images.pitch, cb->hostImageBuf, 
		sizeof(float)*cfg.width, cfg.width*sizeof(float), cfg.height*cb->jobs.size(), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(cb->d_jobs.data, &cb->jobs[0], sizeof(CUDATrackerJob) * cb->jobs.size(), cudaMemcpyHostToDevice);

	cudaEventRecord(cb->imageBufferCopied);
	cb->images.bind(qi_image_texture);
	LocalizeBatchKernel<<<blocks(cb->jobs.size()), threads(), sharedMemSize >>> (cb->jobs.size(), cb->images, kernelParams, cb->d_jobs.data);
	cb->images.unbind(qi_image_texture);
	// Copy back the results
	cudaMemcpyAsync(&cb->jobs[0], cb->d_jobs.data, sizeof(CUDATrackerJob) * cb->jobs.size(), cudaMemcpyDeviceToHost);

	// Make sure we can query the all done signal
	cudaEventRecord(currentBatch->localizationDone);

	active.push_back(currentBatch);
	currentBatch = AllocBatch();
}

void QueuedCUDATracker::Flush()
{
	QueueCurrentBatch();
}

void QueuedCUDATracker::FetchResults()
{
	auto i = active.begin();
	
	while (i != active.end())
	{
		auto cur = i++;
		Batch* b = *cur;

		cudaError_t result = cudaEventQuery(b->localizationDone);
		if (result == cudaSuccess) {
			CopyBatchResults(b);
			active.erase(cur);
			freeBatches.push_back(b);
		}
	}
}

void QueuedCUDATracker::CopyBatchResults(Batch *b)
{
	for (int a=0;a<b->jobs.size();a++) {
		auto j = b->jobs[a];

		LocalizationResult r;
		r.error = j.error;
		r.id = j.id;
		r.firstGuess.x = j.firstGuess.x; r.firstGuess.y = j.firstGuess.y;
		r.locType = j.locType;
		r.zlutIndex = j.zlut;
		r.pos.x = j.resultPos.x;
		r.pos.y = j.resultPos.y;
		r.z = j.resultPos.z;

		results.push_back(r);
	}

	b->jobs.clear();
}

int QueuedCUDATracker::PollFinished(LocalizationResult* dstResults, int maxResults)
{
	FetchResults();
	int numResults = 0;
	while (numResults < maxResults && !results.empty()) {
		dstResults[numResults++] = results.back();
		results.pop_back();
	}
	return numResults;
}

// data can be zero to allocate ZLUT data
void QueuedCUDATracker::SetZLUT(float* data,  int numLUTs, int planes, int res) 
{
	zlut_planes = planes;
	zlut_count = numLUTs;
	zlut_res = res;

	zlut = cudaImageListf::alloc(res, planes, numLUTs);
	if (data) zlut.copyFrom(data, false);
}

// delete[] memory afterwards
float* QueuedCUDATracker::GetZLUT(int *count, int* planes, int *res)
{
	float* data = new float[zlut_planes * zlut_res * zlut_count];
	if (zlut.data)
		zlut.copyTo(data, false);
	else
		std::fill(data, data+(zlut_res*zlut_planes*zlut_count), 0.0f);

	if (planes) *planes = zlut_planes;
	if (res) *res = zlut_res;
	if (count) *count = zlut_count;
	return data;
}




int QueuedCUDATracker::GetResultCount()
{
	FetchResults();
	return results.size();
}

