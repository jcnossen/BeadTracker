/*
Quadrant Interpolation on CUDA

Method:

-Batch images into host-side buffer
-Running batch:
	- Async copy host-side buffer to device
	- Bind image
	- Run COM kernel
	- QI loop: {
		- Run QI kernel: Sample from texture into quadrant profiles
		- Run CUFFT
		- Run QI kernel: Compute positions
	}
	- Async copy results to host
	- Unbind image

*/

#include "std_incl.h"
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_types.h"
#include <cstdint>
#include "utils.h"

#include "QueuedCUDATracker.h"
#include "simplefft.h"
#include "gpu_utils.h"

#define LSQFIT_FUNC __device__ __host__
#include "LsqQuadraticFit.h"

// Types used by QI algorithm
typedef float qivalue_t;
typedef sfft::complex<qivalue_t> qicomplex_t;


// According to this, textures bindings can be switched after the asynchronous kernel is launched
// https://devtalk.nvidia.com/default/topic/392245/texture-binding-and-stream/
texture<float, cudaTextureType2D, cudaReadModeElementType> qi_image_texture(0, cudaFilterModeLinear);


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
	Funcname##Kernel <<<blocks(images.count), threads(), sharedMemPerThread * numThreads >>> (images,param, sharedMemPerThread); \
}



QueuedTracker* CreateQueuedTracker(QTrkSettings* cfg)
{
	return new QueuedCUDATracker(cfg);
}

void CheckCUDAError()
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		const char* errstr = cudaGetErrorString(err);
		dbgprintf("CUDA error: %s\n" ,errstr);
	}
}

QueuedCUDATracker::QueuedCUDATracker(QTrkSettings *cfg, int batchSize)
{
	this->cfg = *cfg;

	// Select the most powerful one
	if (cfg->cuda_device < 0) {
		int numDev;
		cudaGetDeviceCount(&numDev);

		int bestScore;
		int bestDev;
		for (int a=0;a<numDev;a++) {
			int score;
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, a);
			score = prop.multiProcessorCount * prop.clockRate;
			if (a==0 || bestScore < score) {
				bestScore = score;
				bestDev = a;
			}
		}

		cfg->cuda_device = bestDev;
	}

	// We take numThreads to be the number of CUDA streams
	if (cfg->numThreads < 1) {
		cfg->numThreads = 4;
	}

	cudaGetDeviceProperties(&deviceProp, cfg->cuda_device);
	numThreads = deviceProp.warpSize;

	if(batchSize<0) batchSize = 128;
	this->batchSize = batchSize;

	//int sharedSpacePerThread = (prop.sharedMemPerBlock-forward_fft->kparams_size*2) / numThreads;
//	dbgprintf("2X FFT instance requires %d bytes. Space per thread: %d\n", forward_fft->kparams_size*2, sharedSpacePerThread);
	dbgprintf("Device: %s\n", deviceProp.name);
	dbgprintf("Shared memory space:%d bytes. Per thread: %d\n", deviceProp.sharedMemPerBlock, deviceProp.sharedMemPerBlock/numThreads);
	dbgprintf("# of CUDA processors:%d\n", deviceProp.multiProcessorCount);
	dbgprintf("warp size: %d.\n", deviceProp.warpSize);

	qi_FFT_length = 1;
	while (qi_FFT_length < cfg->qi_radialsteps*2) qi_FFT_length *= 2;

	KernelParams &p = kernelParams;
	p.com_bgcorrection = cfg->com_bgcorrection;
	
	ZLUTParams& zp = p.zlut;
	zp.angularSteps = cfg->zlut_angularsteps;
	zp.maxRadius = cfg->zlut_maxradius;
	zp.minRadius = cfg->zlut_minradius;
	zp.planes = zlut_planes;

	QIParams& qi = p.qi;
	qi.angularSteps = cfg->qi_angularsteps;
	qi.iterations = cfg->qi_iterations;
	qi.maxRadius = cfg->qi_maxradius;
	qi.minRadius = cfg->qi_minradius;
	qi.radialSteps = cfg->qi_radialsteps;
	
	zlut = cudaImageListf::empty();
	kernelParams.zlut.img = zlut;

	results.reserve(50000);
	
	streams.resize(cfg->numThreads);
	for (int i=0;i<streams.size();i++) {
		streams[i] = CreateStream();
	}
	currentStream=streams[0];
}

QueuedCUDATracker::~QueuedCUDATracker()
{
	if (zlut.data)
		zlut.free();
	
	DeleteAllElems(streams);
}

static CUBOTH float2 BgCorrectedCOM(int idx, cudaImageListf images, float correctionFactor)
{
	int imgsize = images.w*images.h;
	float sum=0, sum2=0;
	float momentX=0;
	float momentY=0;
//	const float* __restrict__ imgptr = &images.data[imgsize*idx];

	for (int y=0;y<images.h;y++)
		for (int x=0;x<images.w;x++) {
			//float v = imgptr[y*images.pitch+x];
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
			// float v = imgptr[y*images.pitch+x];// images.pixel(x,y,idx);
			float v = images.pixel(x,y,idx);
			v = fabsf(v-mean)-correctionFactor*stdev;
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

__global__ void BgCorrectedCOM(int count, cudaImageListf images,float3* d_com, float bgCorrectionFactor) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < count) {
		float2 com = BgCorrectedCOM(idx, images, bgCorrectionFactor);
		d_com[idx] = make_float3(com.x,com.y,0.0f);
	}
}

static __device__ void RadialProfile(int idx, cudaImageListf& images, float *dst, const ZLUTParams zlut, float2 center, bool& error)
{
	int radialSteps = zlut.img.w;
	for (int i=0;i<zlut.img.w;i++)
		dst[i]=0.0f;

	float totalrmssum2 = 0.0f;
	float rstep = (zlut.maxRadius-zlut.minRadius) / radialSteps;
	for (int i=0;i<radialSteps; i++) {
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
	double invTotalrms = 1.0f/sqrt(totalrmssum2/radialSteps);
	for (int i=0;i<radialSteps;i++) {
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

/*
static __device__ float2 ComputeQIPosition(int idx, cudaImageListf& images, KernelParams params, float2 initial, bool& error)
{
	// Prepare shared memory
	int pl = params.qi.radialSteps*2;
	
	// Localize
	QIParams& qp = params.qi;
	int nr=qp.radialSteps;
	float2 center = initial;
	float pixelsPerProfLen = (qp.maxRadius-qp.minRadius)/qp.radialSteps;

	size_t total_required = sizeof(qivalue_t)*pl*2 + sizeof(qicomplex_t)*pl*2;
	error = false;

	qivalue_t* buf = (qivalue_t*)&params.buffer[total_required * idx];
	qivalue_t* q0=buf, *q1=buf+nr, *q2=buf+nr*2, *q3=buf+nr*3; // buf is sizeof(qivalue_t)*nr*4, or sizeof(qicomplex_t)*nr*2

	qicomplex_t* concat0 = (qicomplex_t*)(buf + nr*4);
	qicomplex_t* concat1 = concat0 + nr;
	qicomplex_t* tmpbuf = concat0 + nr*2;
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

		
		float offsetX = QI_ComputeOffset(concat0, tmpbuf, qp, idx, params.qi.d_twiddles);

		// Build Iy = qB(-r) || qT(r)
		// qT = q0 + q1
		// qB = q2 + q3
		for(int r=0;r<nr;r++) {
			concat0[r] = qicomplex_t(q0[r]+q1[r]);
			concat1[nr-r-1] = qicomplex_t(q2[r]+q3[r]);
		}
		float offsetY = QI_ComputeOffset(concat0, tmpbuf, qp, idx, params.qi.d_twiddles);

		//printf("[%d] OffsetX: %f, OffsetY: %f\n", k, offsetX, offsetY);
		center.x += offsetX * pixelsPerProfLen;
		center.y += offsetY * pixelsPerProfLen;
	}

	//free(buf);
	return center;
}
*/
__global__ void ComputeQIKernel(cudaImageListf images, KernelParams params, float2* d_initial, float2* d_result)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < images.count) {
		bool error;
	//	d_result[idx] = ComputeQIPosition(idx, images, params, d_initial[idx], error);
	}
}


QueuedCUDATracker::Stream::Stream()
{ 
	hostImageBuf = 0; 
	images.data=0; 
	stream=0;
	state = StreamIdle;
	localizeFlags=0;
	jobCount = 0;
}

QueuedCUDATracker::Stream::~Stream() 
{
	cudaStreamDestroy(stream);
	cufftDestroy(fftPlan);

	if(images.data) images.free();
	cudaEventDestroy(localizationDone);
}

bool QueuedCUDATracker::Stream::IsExecutionDone()
{
	return cudaEventQuery(localizationDone) == cudaSuccess;
}

int QueuedCUDATracker::Stream::GetJobCount()
{
	mutex.lock();
	int jc = jobCount;
	mutex.unlock();
	return jc;
}

QueuedCUDATracker::Stream* QueuedCUDATracker::CreateStream()
{
	Stream* s = new Stream();

	cudaStreamCreate(&s->stream);

	uint hostBufSize = sizeof(float)* cfg.width*cfg.height*batchSize;
	s->hostImageBuf.init(hostBufSize);
	s->images = cudaImageListf::alloc(cfg.width, cfg.height, batchSize);

	s->jobs.init(batchSize);
	s->results.init(batchSize);
	s->d_com.init(batchSize);
	s->d_resultpos.init(batchSize);
	s->results.init(batchSize);
	s->d_jobs.init(batchSize);
	s->d_QIprofiles.init(batchSize*2*qi_FFT_length);

	// 2* batchSize, since X & Y both need an FFT transform
	cufftResult_t r = cufftPlanMany(&s->fftPlan, 1, &qi_FFT_length, 
			0, 1, qi_FFT_length, 0, 1, qi_FFT_length, CUFFT_R2C, batchSize*2);

	if(r != CUFFT_SUCCESS) {
		throw std::runtime_error( SPrintf("CUFFT plan creation failed. FFT len: %d. Batchsize: %d\n", qi_FFT_length, batchSize*2));
	}
	cufftSetStream(s->fftPlan, s->stream);

	cudaEventCreate(&s->localizationDone);
	return s;
}

 // get a stream that not currently executing, and still has room for images
QueuedCUDATracker::Stream* QueuedCUDATracker::GetReadyStream()
{
	if (currentStream && currentStream->state != Stream::StreamExecuting && 
		currentStream->GetJobCount() < batchSize) {
		return currentStream;
	}

	// Find another stream that is ready
	while (true) {
		FetchResults();
		for (int a=0;a<streams.size();a++) {
			Stream *s = streams[a];
			if (s->state != Stream::StreamExecuting) {
				currentStream = s;
				dbgprintf("Switching to stream %d\n", a);
				return s;
			}
		}
		Threads::Sleep(5);
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

// All streams on StreamIdle?
bool QueuedCUDATracker::IsIdle()
{
	return CheckAllStreams(Stream::StreamIdle);
}

bool QueuedCUDATracker::CheckAllStreams(Stream::State s)
{
	FetchResults();
	for (int a=0;a<streams.size();a++){
		if (streams[a]->state != s)
			return false;
	}
	return true;
}

bool QueuedCUDATracker::IsQueueFilled()
{
	return CheckAllStreams(Stream::StreamExecuting);
}

bool QueuedCUDATracker::ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane)
{
	Stream* s = GetReadyStream();

	s->lock();

	int jobIndex =  s->jobCount++;
	CUDATrackerJob& job = s->jobs[jobIndex];
	if (initialPos)
		job.initialPos = *(float3*)initialPos;
	job.id = id;
	job.zlut = zlutIndex;
	job.locType = locType;
	job.zlutPlane = zlutPlane;
	s->localizeFlags |= locType; // which kernels to run

	// Copy the image to the batch image buffer (CPU side)
	float* hostbuf = &s->hostImageBuf[cfg.height*cfg.width*jobIndex];
	uchar* srcptr = data;
	CopyImageToFloat(data, cfg.width, cfg.height, pitch, pdt, hostbuf);

	// If batch is filled, copy the image to video memory asynchronously, and start the localization
	if (s->jobCount == batchSize)
		ExecuteBatch(s);

	s->unlock();

	return true;
}

void QueuedCUDATracker::QI_Iterate(device_vec<float3>* initial, device_vec<float3>* newpos, Stream *s)
{
	int njobs = s->jobs.size();
	int nElem = njobs * qi_FFT_length * 2; // 2 profiles of qi_FFT_length size for each job
	//QI_ComputeProfile <<< blocks(nElem), threads(), 0, s->stream >>> (qi_FFT_length, s->
}


__global__ void BuildZLUTKernel(int njobs, cudaImageListf images, ZLUTParams params, float3* positions, CUDATrackerJob* jobs)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < njobs) {
		CUDATrackerJob& j = jobs[idx];
		if (j.locType & LocalizeBuildZLUT) {
			bool err;
			RadialProfile(idx, images, params.GetZLUT( j.zlut, j.zlutPlane ), params, make_float2(positions[idx].x, positions[idx].y), err);
		}
	}
}

void QueuedCUDATracker::ExecuteBatch(Stream *s)
{
	if (s->jobCount==0)
		return;
#ifdef _DEBUG
	dbgprintf("Sending %d images to GPU...\n", s->jobs.size());
#endif

/*	- Async copy host-side buffer to device
	- Bind image
	- Run COM kernel
	- QI loop: {
		- Run QI kernel: Sample from texture into quadrant profiles
		- Run CUFFT
		- Run QI kernel: Compute positions
	}
	- Async copy results to host
	- Unbind image
	*/

//	cb->images.bind(qi_image_texture);
	cudaMemcpy2DAsync( s->images.data, s->images.pitch, s->hostImageBuf.data(), sizeof(float)*s->images.w, s->images.w*sizeof(float), s->images.h * s->jobCount, cudaMemcpyHostToDevice, s->stream);
	s->d_jobs.copyToDevice(s->jobs.data(), s->jobCount, true, s->stream);
	BgCorrectedCOM <<< blocks(s->jobCount), threads(), 0, s->stream >>> (s->jobCount, s->images, s->d_com.data, cfg.com_bgcorrection);

	device_vec<float3> *curpos = &s->d_com;
	for (int a=0;a<cfg.qi_iterations;a++) {
		QI_Iterate(curpos, &s->d_resultpos, s);
		curpos = &s->d_resultpos;
	}

	if (s->localizeFlags & LocalizeBuildZLUT) {
		BuildZLUTKernel <<< blocks(s->jobs.size()), threads(), 0, s->stream >>> (s->jobCount, s->images, kernelParams.zlut, curpos->data, s->d_jobs.data);
	}
	
	curpos->copyToHost(s->results.data(), true, s->stream);
	
//	cb->images.bind(qi_image_texture);
	//CheckCUDAError();
	
	// Make sure we can query the all done signal
	cudaEventRecord(s->localizationDone);

	s->state = Stream::StreamExecuting;
}

void QueuedCUDATracker::Flush()
{
	if (currentStream) {
		currentStream->lock();
		ExecuteBatch(currentStream);
		currentStream->unlock();
	}
}

int QueuedCUDATracker::FetchResults()
{
	// Labview can call from multiple threads
	for (int a=0;a<streams.size();a++) {
		Stream* s = streams[a];
		if (s->state == Stream::StreamExecuting && s->IsExecutionDone()) {
			s->lock();
			CopyStreamResults(s);
			s->state = Stream::StreamIdle;
			s->unlock();
		}
	}
	return results.size();
}

void QueuedCUDATracker::CopyStreamResults(Stream *s)
{
	for (int a=0;a<s->jobCount;a++) {
		CUDATrackerJob& j = s->jobs[a];

		LocalizationResult r;
		r.error = j.error;
		r.id = j.id;
		r.firstGuess = vector2f();
		r.locType = j.locType;
		r.zlutIndex = j.zlut;
		r.pos.x = s->results[a].x;
		r.pos.y = s->results[a].y;
		r.z = s->results[a].z;

		results.push_back(r);
	}
	s->jobCount=0;
	s->localizeFlags = 0; // reset this for the next batch
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
void QueuedCUDATracker::SetZLUT(float* data,  int numLUTs, int planes, int res, float* zcmp) 
{
	zlut_planes = planes;
	zlut_count = numLUTs;
	zlut_res = res;

	if (zcmp) {
		zcompareWindow.copyToDevice(zcmp, res, false);
		kernelParams.zlut.zcmpwindow = zcompareWindow.data;
	}

	zlut = cudaImageListf::alloc(res, planes, numLUTs);
	if (data) zlut.copyToDevice(data, false);
	kernelParams.zlut.img = zlut;
}

// delete[] memory afterwards
float* QueuedCUDATracker::GetZLUT(int *count, int* planes, int *res)
{
	float* data = new float[zlut_planes * zlut_res * zlut_count];
	if (zlut.data)
		zlut.copyToHost(data, false);
	else
		std::fill(data, data+(zlut_res*zlut_planes*zlut_count), 0.0f);

	if (planes) *planes = zlut_planes;
	if (res) *res = zlut_res;
	if (count) *count = zlut_count;

	return data;
}


int QueuedCUDATracker::GetResultCount()
{
	return FetchResults();
}



// TODO: Let GPU copy frames from frames to GPU 
void QueuedCUDATracker::ScheduleFrame(uchar *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, 
									LocalizeType locType, uint frame, uint zlutPlane, bool async)
{
	uchar* img = (uchar*)imgptr;
	int bpp = sizeof(float);
	if (pdt == QTrkU8) bpp = 1;
	else if (pdt == QTrkU16) bpp = 2;
	for (int i=0;i<numROI;i++){
		uchar *roiptr = &img[pitch * positions[i].y + positions[i].x * bpp];
		ScheduleLocalization(roiptr, pitch, pdt, locType, frame, 0, i, zlutPlane);
	}
}

void QueuedCUDATracker::WaitForScheduleFrame(uchar* imgptr) {
}
