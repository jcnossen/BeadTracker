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
		- Run CUFFT. Each iteration per axis does 2x forward FFT, and 1x backward FFT.
		- Run QI kernel: Compute positions
	}
	- Async copy results to host
	- Unbind image


Issues:
- Due to FPU operations on texture coordinates, there are small numerical differences between localizations of the same image at a different position in the batch
- 
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

// Do CPU-side profiling of kernel launches?
//#define TRK_PROFILE

#ifdef TRK_PROFILE
	class ProfileBlock
	{
		double start;
		const char *name;

	public:
		typedef std::pair<int, double> Item;
		static std::map<const char*, Item> results;

		ProfileBlock(const char* name) : name (name) {
			start = GetPreciseTime();
		}
		~ProfileBlock() {
			double end = GetPreciseTime();
			//dbgprintf("%s took %.2f ms\n", name, (end-start)*1000.0f);
			if (results.find(name) == results.end())
				results[name] = Item(1, end-start);
			else {
				Item prev = results[name];
				results[name] = Item (prev.first+1, end-start + prev.second);
			}
		}
	};
	QueuedCUDATracker::ProfileResults ProfileBlock::results;
	QueuedCUDATracker::ProfileResults QueuedCUDATracker::GetProfilingResults() { return ProfileBlock::results; };
#else
	class ProfileBlock {
	public:
		ProfileBlock(const char *name) {}
	};
	QueuedCUDATracker::ProfileResults QueuedCUDATracker::GetProfilingResults() { return QueuedCUDATracker::ProfileResults(); };
#endif


#define LSQFIT_FUNC __device__ __host__
#include "LsqQuadraticFit.h"

// Types used by QI algorithm
typedef float qivalue_t;
typedef sfft::complex<qivalue_t> qicomplex_t;

static std::vector<int> cudaDeviceList; 
void SetCUDADevices(std::vector<int> devices) {
	cudaDeviceList = devices;
}

// According to this, textures bindings can be switched after the asynchronous kernel is launched
// https://devtalk.nvidia.com/default/topic/392245/texture-binding-and-stream/
texture<float, cudaTextureType2D, cudaReadModeElementType> qi_image_texture(0,  cudaFilterModeLinear); // Un-normalized


class PixelSampler_MemCopy {
public:
	// All interpolated texture/images fetches go through here
	static __device__ float Interpolated(cudaImageListf& images, float x,float y, int img, float imgmean)
	{
		return images.interpolate(x,y,img, imgmean);
	}

	// Assumes pixel is always within image bounds
	static __device__ float Index(cudaImageListf& images, int x,int y, int img)
	{
		return images.pixel(x, y, img);
	}
};

class PixelSampler_TextureRead {
public:
	// All interpolated texture/images fetches go through here
	static __device__ float Interpolated(cudaImageListf& images, float x,float y, int img, float imgmean)
	{
		float v;
		if (x < 0 || x > images.w-1 || y < 0 || y > images.h-1)
			v = imgmean;
		else 
			v = tex2D(qi_image_texture, x,y + img*images.h);
		return v;
	}

	// Assumes pixel is always within image bounds
	static __device__ float Index(cudaImageListf& images, int x,int y, int img)
	{
		return tex2D(qi_image_texture, x,y + img*images.h);
	}
};


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

static int GetBestCUDADevice()
{
	int bestScore;
	int bestDev;
	int numDev;
	cudaGetDeviceCount(&numDev);
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
	return bestDev;
}

void QueuedCUDATracker::InitializeDeviceList()
{
	int numDevices;
	cudaGetDeviceCount(&numDevices);

	// Select the most powerful one
	if (cfg.cuda_device == QTrkCUDA_UseBest) {
		cfg.cuda_device = GetBestCUDADevice();
		devices.push_back(new Device(cfg.cuda_device));
	} else if(cfg.cuda_device == QTrkCUDA_UseAll) {
		// Use all devices
		for (int i=0;i<numDevices;i++)
			devices.push_back(new Device(i));
	} else if (cfg.cuda_device == QTrkCUDA_UseList) {
		for (int i=0;i<cudaDeviceList.size();i++)
			devices.push_back(new Device(cudaDeviceList[i]));
	} else {
		devices.push_back (new Device(cfg.cuda_device));
	}
}


QueuedCUDATracker::QueuedCUDATracker(QTrkSettings *cfg, int batchSize)
{
	this->cfg = *cfg;

	InitializeDeviceList();

	// We take numThreads to be the number of CUDA streams
	if (cfg->numThreads < 1) {
		cfg->numThreads = devices.size()*3;
	}
	int numStreams = cfg->numThreads;

	cudaGetDeviceProperties(&deviceProp, devices[0]->index);
	numThreads = deviceProp.warpSize;
	
	if(batchSize<0) batchSize = 512;
	while (batchSize * cfg->height > deviceProp.maxTexture2D[1]) {
		batchSize/=2;
	}
	this->batchSize = batchSize;

	qi_FFT_length = 1;
	while (qi_FFT_length < cfg->qi_radialsteps*2) qi_FFT_length *= 2;

	//int sharedSpacePerThread = (prop.sharedMemPerBlock-forward_fft->kparams_size*2) / numThreads;
//	dbgprintf("2X FFT instance requires %d bytes. Space per thread: %d\n", forward_fft->kparams_size*2, sharedSpacePerThread);
	dbgprintf("Device: %s.\n", deviceProp.name);
	//dbgprintf("Shared memory space:%d bytes. Per thread: %d\n", deviceProp.sharedMemPerBlock, deviceProp.sharedMemPerBlock/numThreads);
	dbgprintf("# of CUDA processors:%d. Using %d streams\n", deviceProp.multiProcessorCount, numStreams);
	dbgprintf("Warp size: %d. Max threads: %d, Batch size: %d. QI FFT Length: %d\n", deviceProp.warpSize, deviceProp.maxThreadsPerBlock, batchSize, qi_FFT_length);

	KernelParams &p = kernelParams;
	p.com_bgcorrection = cfg->com_bgcorrection;
	
	ZLUTParams& zp = p.zlut;
	zp.angularSteps = cfg->zlut_angularsteps;
	zp.maxRadius = cfg->zlut_maxradius;
	zp.minRadius = cfg->zlut_minradius;
	zp.planes = 0;
	zp.zcmpwindow = 0;

	QIParams& qi = p.qi;
	qi.angularSteps = cfg->qi_angsteps_per_quadrant;
	qi.iterations = cfg->qi_iterations;
	qi.maxRadius = cfg->qi_maxradius;
	qi.minRadius = cfg->qi_minradius;
	qi.radialSteps = cfg->qi_radialsteps;
	std::vector<float2> qi_radialgrid(qi.angularSteps);
	for (int i=0;i<qi.angularSteps;i++)  {
		float ang = 0.5f*3.141593f*i/(float)qi.angularSteps;
		qi_radialgrid[i]=make_float2(cos(ang), sin(ang));
	}

	std::vector<float2> zlut_radialgrid(cfg->zlut_angularsteps);
	for (int i=0;i<cfg->zlut_angularsteps;i++) {
		float ang = 2*3.141593f*i/(float)cfg->zlut_angularsteps;
		zlut_radialgrid[i]=make_float2(cos(ang),sin(ang));
	}

	for (int i=0;i<devices.size();i++) {
		Device* d = devices[i];
		cudaSetDevice(d->index);
		d->d_qiradialgrid=qi_radialgrid;
		d->d_zlutradialgrid = zlut_radialgrid;
	}
	kernelParams.zlut.img = cudaImageListf::empty();
	
	streams.reserve(numStreams);
	try {
		for (int i=0;i<numStreams;i++)
			streams.push_back( CreateStream( devices[i%devices.size()] ) );
	}
	catch(...) {
		DeleteAllElems(streams);
		throw;
	}

	currentStream=streams[0];
	int memUsePerStream = streams[0]->CalcMemoryUse();
	dbgprintf("Stream memory use: %d kb", memUsePerStream/1024);

	batchesDone = 0;
	time_QI = time_COM = time_ZCompute = time_imageCopy = 0.0;
	useTextureCache = false;
}

QueuedCUDATracker::~QueuedCUDATracker()
{
	DeleteAllElems(streams);
	DeleteAllElems(devices);	
}

QueuedCUDATracker::Device::~Device()
{
	cudaSetDevice(index);
	zlut.free();
}

template<typename TImageSampler>
__device__ float2 BgCorrectedCOM(int idx, cudaImageListf images, float correctionFactor, float* pMean)
{
	int imgsize = images.w*images.h;
	float sum=0, sum2=0;
	float momentX=0;
	float momentY=0;

	for (int y=0;y<images.h;y++)
		for (int x=0;x<images.w;x++) {
			float v = TImageSampler::Index(images, x, y, idx);
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
			float v = TImageSampler::Index(images, x,y,idx);
			v = fabsf(v-mean)-correctionFactor*stdev;
			if(v<0.0f) v=0.0f;
			sum += v;
			momentX += x*v;
			momentY += y*v;
		}

	if (pMean)
		*pMean = mean;

	float2 com;
	com.x = momentX / (float)sum;
	com.y = momentY / (float)sum;
	return com;
}

template<typename TImageSampler>
__global__ void BgCorrectedCOM(int count, cudaImageListf images,float3* d_com, float* d_means, float bgCorrectionFactor) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < count) {
		float mean;
		float2 com = BgCorrectedCOM<TImageSampler> (idx, images, bgCorrectionFactor, &mean);

		d_means[idx] = mean;
		d_com[idx] = make_float3(com.x,com.y,0.0f);
	}
}

QueuedCUDATracker::Stream::Stream()
{ 
	device = 0;
	hostImageBuf = 0; 
	images.data=0; 
	stream=0;
	state = StreamIdle;
	localizeFlags=0;
}

QueuedCUDATracker::Stream::~Stream() 
{
	cudaSetDevice(device->index);
	cufftDestroy(fftPlan);

	if(images.data) images.free();
	cudaEventDestroy(localizationDone);
	cudaEventDestroy(qiDone);
	cudaEventDestroy(comDone);
	cudaEventDestroy(imageCopyDone);
	cudaEventDestroy(zcomputeDone);
	cudaEventDestroy(batchStart);

	if (stream)
		cudaStreamDestroy(stream); // stream can be zero if in debugStream mode.
}



bool QueuedCUDATracker::Stream::IsExecutionDone()
{
	cudaSetDevice(device->index);
	return cudaEventQuery(localizationDone) == cudaSuccess;
}

int QueuedCUDATracker::Stream::CalcMemoryUse()
{
	return d_com.memsize() + d_zlutmapping.memsize() + d_QIprofiles.memsize() + d_QIprofiles_reverse.memsize() + d_radialprofiles.memsize() + d_imgmeans.memsize() +
		d_quadrants.memsize() + d_resultpos.memsize() + d_zlutcmpscores.memsize();
}


QueuedCUDATracker::Stream* QueuedCUDATracker::CreateStream(Device* device)
{
	Stream* s = new Stream();

	try {
		s->device = device;
		cudaSetDevice(device->index);
		cudaStreamCreate(&s->stream);

		uint hostBufSize = cfg.width*cfg.height*batchSize;
		s->hostImageBuf.init(hostBufSize);
		s->images = cudaImageListf::alloc(cfg.width, cfg.height, batchSize);

		s->jobs.reserve(batchSize);
		s->results.init(batchSize);
		s->com.init(batchSize);
		s->d_com.init(batchSize);
		s->d_resultpos.init(batchSize);
		s->results.init(batchSize);
		s->zlutmapping.init(batchSize);
		s->d_zlutmapping.init(batchSize);
		s->d_quadrants.init(qi_FFT_length*batchSize*2);
		s->d_QIprofiles.init(batchSize*2*qi_FFT_length); // (2 axis) * (2 radialsteps) = 8 * nr = 2 * qi_FFT_length
		s->d_QIprofiles_reverse.init(batchSize*2*qi_FFT_length);
		s->d_radialprofiles.init(cfg.zlut_radialsteps*batchSize);
		s->d_imgmeans.init(batchSize);
	
		// 2* batchSize, since X & Y both need an FFT transform
		//cufftResult_t r = cufftPlanMany(&s->fftPlan, 1, &qi_FFT_length, 0, 1, qi_FFT_length, 0, 1, qi_FFT_length, CUFFT_C2C, batchSize*4);
		cufftResult_t r = cufftPlan1d(&s->fftPlan, qi_FFT_length, CUFFT_C2C, batchSize*2);

		if(r != CUFFT_SUCCESS) {
			throw std::runtime_error( SPrintf("CUFFT plan creation failed. FFT len: %d. Batchsize: %d\n", qi_FFT_length, batchSize*4));
		}
		cufftSetCompatibilityMode(s->fftPlan, CUFFT_COMPATIBILITY_NATIVE);
		cufftSetStream(s->fftPlan, s->stream);

		cudaEventCreate(&s->localizationDone);
		cudaEventCreate(&s->comDone);
		cudaEventCreate(&s->imageCopyDone);
		cudaEventCreate(&s->zcomputeDone);
		cudaEventCreate(&s->qiDone);
		cudaEventCreate(&s->batchStart);
	} catch (const std::exception& e) {
		delete s;
		throw;
	}
	return s;
}

 // get a stream that not currently executing, and still has room for images
QueuedCUDATracker::Stream* QueuedCUDATracker::GetReadyStream()
{
	if (currentStream && currentStream->state != Stream::StreamExecuting && 
		currentStream->jobs.size() < batchSize) {
		return currentStream;
	}

	// Find another stream that is ready
	// First round: Check streams with current non-updated state. 
	// Second round: Query the GPU again for updated stream state.
	// Further rounds: Wait 1 ms and try again.
	for (int i = 0; true; i ++) {
		for (int a=0;a<streams.size();a++) {
			Stream *s = streams[a];
			if (s->state != Stream::StreamExecuting) {
				currentStream = s;
				dbgprintf("Switching to stream %d\n", a);
				return s;
			}
		}
		FetchResults();
		if (i > 0) Threads::Sleep(1);
	}
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

void QueuedCUDATracker::ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob* jobInfo )
{
	Stream* s = GetReadyStream();
	s->lock();

	int jobIndex = s->jobs.size();
	s->jobs.push_back(*jobInfo);
	s->localizeFlags |= jobInfo->LocType(); // which kernels to run
	s->zlutmapping[jobIndex].locType = jobInfo->LocType();
	s->zlutmapping[jobIndex].zlutIndex = jobInfo->zlutIndex;
	s->zlutmapping[jobIndex].zlutPlane = jobInfo->zlutPlane;

	// Copy the image to the batch image buffer (CPU side)
	float* hostbuf = &s->hostImageBuf[cfg.height*cfg.width*jobIndex];
	CopyImageToFloat(data, cfg.width, cfg.height, pitch, pdt, hostbuf);

//	tmp = floatToNormalizedInt( (float*)hostbuf, cfg.width,cfg.height,(uchar)255);
//	WriteJPEGFile(tmp, cfg.width,cfg.height, "writehostbuf2.jpg", 99);
//	delete[] tmp;

	// If batch is filled, copy the image to video memory asynchronously, and start the localization
	if (s->jobs.size() == batchSize) {
		if (useTextureCache)
			ExecuteBatch<PixelSampler_TextureRead> (s);
		else
			ExecuteBatch<PixelSampler_MemCopy> (s);
	}

	s->unlock();
}


void QueuedCUDATracker::Flush()
{
	if (currentStream && currentStream->state == Stream::StreamIdle) {
		currentStream->lock();

		if (useTextureCache) ExecuteBatch<PixelSampler_TextureRead> (currentStream);
		else ExecuteBatch<PixelSampler_MemCopy> (currentStream);
		currentStream->unlock();
		currentStream = 0;
	}
}

template<typename T>
static __device__ T interpolate(T a, T b, float x) { return a + (b-a)*x; }

template<typename TImageSampler>
__device__ void ComputeQuadrantProfile(cudaImageListf& images, int idx, float* dst, const QIParams& params, int quadrant, float img_mean, float2 center)
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
	
	float sum2=0.0f;
	float total = 0.0f;
	float rstep = (params.maxRadius - params.minRadius) / params.radialSteps;
	for (int i=0;i<params.radialSteps; i++) {
		float sum = 0.0f;
		float r = params.minRadius + rstep * i;

		for (int a=0;a<params.angularSteps;a++) {
			float x = center.x + mx*params.radialgrid[a].x * r;
			float y = center.y + my*params.radialgrid[a].y * r;
			float v = TImageSampler::Interpolated(images, x,y, idx, img_mean);
			sum += v;
		}

		dst[i] = sum/params.angularSteps - img_mean;
		total += dst[i];
		sum2+=sum;
	}
}

template<typename TImageSampler>
__global__ void QI_ComputeProfile(int count, cudaImageListf images, float3* positions, float* quadrants, float2* profiles, float2* reverseProfiles, float* img_means, QIParams params)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < count) {
		int fftlen = params.radialSteps*2;
		float* img_qdr = &quadrants[ idx * params.radialSteps * 4 ];
		for (int q=0;q<4;q++)
			ComputeQuadrantProfile<TImageSampler> (images, idx, &img_qdr[q*params.radialSteps], params, q, img_means[idx], make_float2(positions[idx].x, positions[idx].y));

		int nr = params.radialSteps;
		qicomplex_t* imgprof = (qicomplex_t*) &profiles[idx * fftlen*2];
		qicomplex_t* x0 = imgprof;
		qicomplex_t* x1 = imgprof + nr*1;
		qicomplex_t* y0 = imgprof + nr*2;
		qicomplex_t* y1 = imgprof + nr*3;

		qicomplex_t* revprof = (qicomplex_t*)&reverseProfiles[idx*fftlen*2];
		qicomplex_t* xrev = revprof;
		qicomplex_t* yrev = revprof + nr*2;

		float* q0 = &img_qdr[0];
		float* q1 = &img_qdr[nr];
		float* q2 = &img_qdr[nr*2];
		float* q3 = &img_qdr[nr*3];

		// Build Ix = qL(-r) || qR(r)
		// qL = q1 + q2   (concat0)
		// qR = q0 + q3   (concat1)
		for(int r=0;r<nr;r++) {
			x0[nr-r-1] = qicomplex_t(q1[r]+q2[r]);
			x1[r] = qicomplex_t(q0[r]+q3[r]);
		}
		// Build Iy = [ qB(-r)  qT(r) ]
		// qT = q0 + q1
		// qB = q2 + q3
		for(int r=0;r<nr;r++) {
			y1[r] = qicomplex_t(q0[r]+q1[r]);
			y0[nr-r-1] = qicomplex_t(q2[r]+q3[r]);
		}


		for(int r=0;r<nr*2;r++)
			xrev[r] = x0[nr*2-r-1];
		for(int r=0;r<nr*2;r++)
			yrev[r] = y0[nr*2-r-1];
	}
}


__global__ void QI_MultiplyWithConjugate(int n, cufftComplex* a, cufftComplex* b)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		cufftComplex A = a[idx];
		cufftComplex B = b[idx];
	
		a[idx] = make_float2(A.x*B.x + A.y*B.y, A.y*B.x -A.x*B.y); // multiplying with conjugate
	}
}

__shared__ float compute_offset_buffer[];

__device__ float QI_ComputeAxisOffset(cufftComplex* autoconv, int fftlen)
{
	typedef float compute_t;
	//compute_t* shifted = (compute_t*)malloc(fftlen*sizeof(compute_t)); 
	float* shifted = &compute_offset_buffer [threadIdx.x * fftlen];
	int nr = fftlen/2;
	for(int x=0;x<fftlen;x++)  {
		shifted[x] = autoconv[(x+nr)%(nr*2)].x;
	}

	compute_t maxPos = ComputeMaxInterp<compute_t>::Compute(shifted, fftlen);
	compute_t offset = (maxPos - nr) / (3.14159265359f * 0.5f);
	//free(shifted);
	return offset;
}

__global__ void QI_OffsetPositions(int njobs, float3* current, float3* dst, cufftComplex* autoconv, int fftLength, float2* offsets, float pixelsPerProfLen)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < njobs) {
		// X
		cufftComplex* autoconvX = &autoconv[idx * fftLength * 2];
		float xoffset = QI_ComputeAxisOffset(autoconvX, fftLength);

		cufftComplex* autoconvY = autoconvX + fftLength;
		float yoffset = QI_ComputeAxisOffset(autoconvY, fftLength);

		dst[idx].x = current[idx].x + xoffset * pixelsPerProfLen;
		dst[idx].y = current[idx].y + yoffset * pixelsPerProfLen;

		if (offsets) 
			offsets[idx] = make_float2( xoffset, yoffset);
	}
}



/*
		q0: xprof[r], yprof[r]
		q1: xprof[len-r-1], yprof[r]
		q2: xprof[len-r-1], yprof[len-r-1]
		q3: xprof[r], yprof[len-r-1]

	kernel gets called with dim3(images.count, radialsteps*4) elements
*/
template<typename TImageSampler>
__global__ void QI_ComputeQuadrants(int njobs, cudaImageListf images, float3* positions, float* dst_quadrants, float* imgmeans, const QIParams params)
{
	int jobIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int rIdx = threadIdx.y + blockIdx.y * blockDim.y;

	if (jobIdx < njobs && rIdx < 4*params.radialSteps) {

		const int qmat[] = {
			1, 1,
			-1, 1,
			-1, -1,
			1, -1 };

		int quadrant = rIdx & 3;
		int mx = qmat[2*quadrant+0];
		int my = qmat[2*quadrant+1];
		float* qdr = &dst_quadrants[ (jobIdx * 4 + quadrant) * params.radialSteps ];

		float rstep = (params.maxRadius - params.minRadius) / params.radialSteps;
		float sum = 0.0f;
		float r = params.minRadius + rstep * rIdx;
		float3 pos = positions[jobIdx];
		float mean = imgmeans[jobIdx];

		for (int a=0;a<params.angularSteps;a++) {
			float x = pos.x + mx*params.radialgrid[a].x * r;
			float y = pos.y + my*params.radialgrid[a].y * r;
			sum += TImageSampler::Interpolated(images, x,y,jobIdx, mean);
		}
		qdr[rIdx] = sum/params.angularSteps-mean;
	}
}

__global__ void QI_QuadrantsToProfiles(int njobs, cudaImageListf images, float* quadrants, float2* profiles, float2* reverseProfiles,  const QIParams params)
{
//ComputeQuadrantProfile(cudaImageListf& images, int idx, float* dst, const QIParams& params, int quadrant, float2 center)
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < njobs) {
		int fftlen = params.radialSteps*2;
		float* img_qdr = &quadrants[ idx * params.radialSteps * 4 ];

		int nr = params.radialSteps;
		qicomplex_t* imgprof = (qicomplex_t*) &profiles[idx * fftlen*2];
		qicomplex_t* x0 = imgprof;
		qicomplex_t* x1 = imgprof + nr*1;
		qicomplex_t* y0 = imgprof + nr*2;
		qicomplex_t* y1 = imgprof + nr*3;

		qicomplex_t* revprof = (qicomplex_t*)&reverseProfiles[idx*fftlen*2];
		qicomplex_t* xrev = revprof;
		qicomplex_t* yrev = revprof + nr*2;

		float* q0 = &img_qdr[0];
		float* q1 = &img_qdr[nr];
		float* q2 = &img_qdr[nr*2];
		float* q3 = &img_qdr[nr*3];

		// Build Ix = qL(-r) || qR(r)
		// qL = q1 + q2   (concat0)
		// qR = q0 + q3   (concat1)
		for(int r=0;r<nr;r++) {
			x0[nr-r-1] = qicomplex_t(q1[r]+q2[r]);
			x1[r] = qicomplex_t(q0[r]+q3[r]);
		}
		// Build Iy = [ qB(-r)  qT(r) ]
		// qT = q0 + q1
		// qB = q2 + q3
		for(int r=0;r<nr;r++) {
			y1[r] = qicomplex_t(q0[r]+q1[r]);
			y0[nr-r-1] = qicomplex_t(q2[r]+q3[r]);
		}

		for(int r=0;r<nr*2;r++)
			xrev[r] = x0[nr*2-r-1];
		for(int r=0;r<nr*2;r++)
			yrev[r] = y0[nr*2-r-1];
	}
}


static unsigned long hash(unsigned char *str, int n)
{
    unsigned long hash = 5381;
    
    for (int i=0;i<n;i++) {
		int c = str[i];
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
	}

    return hash;
}

template<typename T>
void checksum(T* data, int elemsize, int numelem, const char *name)
{
#ifdef QI_DBG_EXPORT
	uchar* cp = (uchar*)ALLOCA(elemsize*numelem*sizeof(T));
	cudaDeviceSynchronize();
	cudaMemcpy(cp, data, sizeof(T)*elemsize*numelem, cudaMemcpyDeviceToHost);

	dbgprintf("%s:\n", name);
	for (int i=0;i<numelem;i++) {
		uchar *elem = cp+elemsize*sizeof(T)*i;
		dbgprintf("[%d]: %d\n", i, hash(elem, elemsize));
	}
#endif
}

template<typename TImageSampler>
void QueuedCUDATracker::QI_Iterate(device_vec<float3>* initial, device_vec<float3>* newpos, Stream *s)
{
	int njobs = s->jobs.size();
	dim3 qdrThreads(16, 16);

	if (0) {
		QI_ComputeQuadrants<TImageSampler> <<< dim3( (njobs + qdrThreads.x - 1) / qdrThreads.x, (4*cfg.qi_radialsteps + qdrThreads.y - 1) / qdrThreads.y ), qdrThreads, 0, s->stream >>> 
			(njobs, s->images, initial->data, s->d_quadrants.data, s->d_imgmeans.data, kernelParams.qi);

		QI_QuadrantsToProfiles <<< blocks(njobs), threads(), 0, s->stream >>> 
			(njobs, s->images, s->d_quadrants.data, s->d_QIprofiles.data, s->d_QIprofiles_reverse.data, kernelParams.qi);
	}
	else {
		QI_ComputeProfile <TImageSampler> <<< blocks(njobs), threads(), 0, s->stream >>> (njobs, s->images, initial->data, 
			s->d_quadrants.data, s->d_QIprofiles.data, s->d_QIprofiles_reverse.data, s->d_imgmeans.data,  kernelParams.qi);
	}
	checksum(s->d_quadrants.data, qi_FFT_length * 2, njobs, "quadrant");
	checksum(s->d_QIprofiles.data, qi_FFT_length * 2, njobs, "prof");
	checksum(s->d_QIprofiles_reverse.data, qi_FFT_length * 2, njobs, "revprof");

	cufftComplex* prof = (cufftComplex*)s->d_QIprofiles.data;
	cufftComplex* revprof = (cufftComplex*)s->d_QIprofiles_reverse.data;

	cufftExecC2C(s->fftPlan, prof, prof, CUFFT_FORWARD);
	cufftExecC2C(s->fftPlan, revprof, revprof, CUFFT_FORWARD);

	int nval = qi_FFT_length * 2 * batchSize, nthread=256;
	QI_MultiplyWithConjugate<<< dim3( (nval + nthread - 1)/nthread ), dim3(nthread), 0, s->stream >>>(nval, prof, revprof);
	cufftExecC2C(s->fftPlan, prof, prof, CUFFT_INVERSE);

	float2* d_offsets=0;
	float pixelsPerProfLen = (cfg.qi_maxradius-cfg.qi_minradius)/cfg.qi_radialsteps;
	QI_OffsetPositions<<<blocks(njobs), threads(), sizeof(float)*qi_FFT_length*numThreads, s->stream>>>
		(njobs, initial->data, newpos->data, prof, qi_FFT_length, d_offsets, pixelsPerProfLen); // revprof is used as temp buffer to shift the autocorrelation profile
}


__global__ void ZLUT_ProfilesToZLUT(int njobs, cudaImageListf images, ZLUTParams params, float3* positions, ZLUTMapping* mapping, float* profiles)
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx < njobs) {
		ZLUTMapping m = mapping[idx];
		if (m.locType & LocalizeBuildZLUT) {
			bool err;
			float* dst = params.GetZLUT(m.zlutIndex, m.zlutPlane );

			for (int i=0;i<params.radialSteps();i++)
				dst [i] = profiles [ params.radialSteps()*idx + i ];
		}
	}
}


// Compute a single ZLUT radial profile element (looping through all the pixels at constant radial distance)
template<typename TImageSampler>
__global__ void ZLUT_RadialProfileKernel(int njobs, cudaImageListf images, ZLUTParams params, float3* positions, float* profiles, float* imgmeans)
{
	int jobIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int radialIdx = threadIdx.y + blockIdx.y * blockDim.y;

	if (jobIdx >= njobs || radialIdx >= params.radialSteps()) 
		return;

	float* dstprof = &profiles[params.radialSteps() * jobIdx];
	float r = params.minRadius + (params.maxRadius-params.minRadius)*radialIdx/params.radialSteps();
	float sum = 0.0f;
	float imgmean = imgmeans[jobIdx];
	
	for (int i=0;i<params.angularSteps;i++) {
		float x = positions[jobIdx].x + params.radialgrid[i].x * r;
		float y = positions[jobIdx].y + params.radialgrid[i].y * r;

		sum += TImageSampler::Interpolated(images, x,y, jobIdx, imgmean);
	}
	dstprof [radialIdx] = sum/params.angularSteps-imgmean;
}


__global__ void ZLUT_ComputeZ (int njobs, ZLUTParams params, float3* positions, float* compareScoreBuf, ZLUTMapping *mapping)
{
	int jobIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (jobIdx < njobs && (mapping[jobIdx].locType & LocalizeZ)) {
		float* cmp = &compareScoreBuf [params.planes * jobIdx];

		float maxPos = ComputeMaxInterp<float>::Compute(cmp, params.planes);
		positions[jobIdx].z = maxPos;
	}
}

__global__ void ZLUT_ComputeProfileMatchScores(int njobs, ZLUTParams params, float *profiles, float* compareScoreBuf, ZLUTMapping* zlutmap)
{
	int jobIdx = threadIdx.x + blockIdx.x * blockDim.x;
	int zPlaneIdx = threadIdx.y + blockIdx.y * blockDim.y;

	if (jobIdx >= njobs || zPlaneIdx >= params.planes)
		return;

	float* prof = &profiles [jobIdx * params.radialSteps()];
	ZLUTMapping mapping = zlutmap[jobIdx];
	if (mapping.locType & LocalizeZ) {
		float diffsum = 0.0f;
		for (int r=0;r<params.radialSteps();r++) {
			float d = prof[r] - params.img.pixel(r, zPlaneIdx, zlutmap[jobIdx].zlutIndex);
			if (params.zcmpwindow)
				d *= params.zcmpwindow[r];
			diffsum += d*d;
		}

		compareScoreBuf[ params.planes * jobIdx + zPlaneIdx ] = -diffsum;
	}
}

__global__ void ZLUT_NormalizeProfiles(int njobs, ZLUTParams params, float* profiles)
{
	int jobIdx = threadIdx.x + blockIdx.x * blockDim.x;

	if (jobIdx < njobs) {
		float* prof = &profiles[params.radialSteps()*jobIdx];
		float rmsSum2 = 0.0f;
		for (int i=0;i<params.radialSteps();i++){
			rmsSum2 += prof[i]*prof[i];
		}
		float invTotalRms = 1.0f / sqrt(rmsSum2/params.radialSteps());
		for (int i=0;i<params.radialSteps();i++)
			prof[i] *= invTotalRms;
	}
}



template<typename TImageSampler>
void QueuedCUDATracker::ExecuteBatch(Stream *s)
{
	if (s->JobCount()==0)
		return;
	//dbgprintf("Sending %d images to GPU stream %p...\n", s->jobCount, s->stream);

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

	Device *d = s->device;
	cudaSetDevice(d->index);
	kernelParams.qi.radialgrid = d->d_qiradialgrid.data;
	kernelParams.zlut.img = d->zlut;
	kernelParams.zlut.radialgrid = d->d_zlutradialgrid.data;
	kernelParams.zlut.zcmpwindow = d->zcompareWindow.data;

	cudaEventRecord(s->batchStart, s->stream);

	{ ProfileBlock p("image to gpu");
	cudaMemcpy2DAsync( s->images.data, s->images.pitch, s->hostImageBuf.data(), sizeof(float)*s->images.w, s->images.w*sizeof(float), s->images.h * s->JobCount(), cudaMemcpyHostToDevice, s->stream); }
	//{ ProfileBlock p("jobs to gpu");
	//s->d_jobs.copyToDevice(s->jobs.data(), s->jobCount, true, s->stream); }
	cudaEventRecord(s->imageCopyDone, s->stream);
	s->images.bind(qi_image_texture);
	{ ProfileBlock p("COM");
	BgCorrectedCOM<TImageSampler> <<< blocks(s->JobCount()), threads(), 0, s->stream >>> 
		(s->JobCount(), s->images, s->d_com.data, s->d_imgmeans.data, cfg.com_bgcorrection);
	checksum(s->d_com.data, 1, s->JobCount(), "com");
	}
	cudaEventRecord(s->comDone, s->stream);

//	{ ProfileBlock p("COM results to host");
	s->d_com.copyToHost(s->com.data(), true, s->stream);

	device_vec<float3> *curpos = &s->d_com;
	if (s->localizeFlags & LocalizeQI) {
		ProfileBlock p("QI");
		for (int a=0;a<cfg.qi_iterations;a++) {
			QI_Iterate<TImageSampler> (curpos, &s->d_resultpos, s);
			curpos = &s->d_resultpos;
		}
	}
	cudaEventRecord(s->qiDone, s->stream);

	// Compute radial profiles
	if (s->localizeFlags & (LocalizeZ | LocalizeBuildZLUT)) {
		dim3 numThreads(16, 16);
		dim3 numBlocks( (s->JobCount() + numThreads.x - 1) / numThreads.x, (cfg.zlut_radialsteps + numThreads.y - 1) / numThreads.y);
		{ ProfileBlock p("ZLUT radial profile");
		ZLUT_RadialProfileKernel<TImageSampler> <<< numBlocks , numThreads, 0, s->stream >>>
			(s->JobCount(), s->images, kernelParams.zlut, curpos->data, s->d_radialprofiles.data,  s->d_imgmeans.data); }

		{ ProfileBlock p("ZLUT normalize profiles");
		ZLUT_NormalizeProfiles<<< blocks(s->JobCount()), threads(), 0, s->stream >>> (s->JobCount(), kernelParams.zlut, s->d_radialprofiles.data); }

		s->d_zlutmapping.copyToDevice(s->zlutmapping.data(), s->JobCount(), true, s->stream);
	}
	// Store profile in LUT
	if (s->localizeFlags & LocalizeBuildZLUT) {
		{ ProfileBlock p("ZLUT build zlut");
		ZLUT_ProfilesToZLUT <<< blocks(s->JobCount()), threads(), 0, s->stream >>> (s->JobCount(), s->images, kernelParams.zlut, curpos->data, s->d_zlutmapping.data, s->d_radialprofiles.data); }
	}
	// Compute Z 
	if (s->localizeFlags & LocalizeZ) {
		int zplanes = kernelParams.zlut.planes;
		dim3 numThreads(8, 16);
		{ ProfileBlock p("ZLUT compute Z");
		ZLUT_ComputeProfileMatchScores <<< dim3( (s->JobCount() + numThreads.x - 1) / numThreads.x, (zplanes  + numThreads.y - 1) / numThreads.y), numThreads, 0, s->stream >>> 
			(s->JobCount(), kernelParams.zlut, s->d_radialprofiles.data, s->d_zlutcmpscores.data, s->d_zlutmapping.data);
		ZLUT_ComputeZ <<< blocks(s->JobCount()), threads(), 0, s->stream >>> (s->JobCount(), kernelParams.zlut, curpos->data, s->d_zlutcmpscores.data, s->d_zlutmapping.data);
		}
	}
	s->images.unbind(qi_image_texture);
	cudaEventRecord(s->zcomputeDone, s->stream);

	{ ProfileBlock p("Results to host");
	curpos->copyToHost(s->results.data(), true, s->stream);}

	// Make sure we can query the all done signal
	cudaEventRecord(s->localizationDone, s->stream);

	s->state = Stream::StreamExecuting;
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
	for (int a=0;a<s->JobCount();a++) {
		LocalizationJob& j = s->jobs[a];

		LocalizationResult r;
		r.job = j;
		r.firstGuess =  vector2f( s->com[a].x, s->com[a].y );
		r.pos = vector3f( s->results[a].x , s->results[a].y, s->results[a].z);

		results.push_back(r);
	}

	// Update times
	float qi, com, imagecopy, zcomp;
	cudaEventElapsedTime(&imagecopy, s->batchStart, s->imageCopyDone);
	cudaEventElapsedTime(&com, s->imageCopyDone, s->comDone);
	cudaEventElapsedTime(&qi, s->comDone, s->qiDone);
	cudaEventElapsedTime(&zcomp, s->qiDone, s->zcomputeDone);
	time_COM += com;
	time_QI += qi;
	time_imageCopy += imagecopy;
	time_ZCompute += zcomp;
	batchesDone ++;
	
	s->jobs.clear();
	s->localizeFlags = 0; // reset this for the next batch
}

int QueuedCUDATracker::PollFinished(LocalizationResult* dstResults, int maxResults)
{
	FetchResults();

	int numResults = 0;
	while (numResults < maxResults && !results.empty()) {
		dstResults[numResults++] = results.front();
		results.pop_front();
	}
	return numResults;
}

// data can be zero to allocate ZLUT data
void QueuedCUDATracker::SetZLUT(float* data,  int numLUTs, int planes, float* zcmp) 
{
	kernelParams.zlut.planes = planes;

	for (int i=0;i<streams.size();i++) {
		cudaSetDevice(streams[i]->device->index);
		streams[i]->d_zlutcmpscores.init(planes * batchSize);
	}

	for (int i=0;i<devices.size();i++) {
		devices[i]->SetZLUT(data, cfg.zlut_radialsteps, planes, numLUTs, zcmp);
	}
}

void QueuedCUDATracker::Device::SetZLUT(float *data, int radialsteps, int planes, int numLUTs, float* zcmp)
{
	cudaSetDevice(index);

	if (zcmp)
		zcompareWindow.copyToDevice(zcmp, radialsteps, false);
	else 
		zcompareWindow.free();

	zlut = cudaImageListf::alloc(radialsteps, planes, numLUTs);
	if (data) zlut.copyToDevice(data, false);
	else zlut.clear();
}	

// delete[] memory afterwards
float* QueuedCUDATracker::GetZLUT(int *count, int* planes)
{
	cudaImageListf* zlut = &devices[0]->zlut;
	float* data = new float[zlut->h * cfg.zlut_radialsteps * zlut->count];
	if (zlut->data)
		zlut->copyToHost(data, false);
	else
		std::fill(data, data+(cfg.zlut_radialsteps*zlut->h*zlut->count), 0.0f);

	if (planes) *planes = zlut->h;
	if (count) *count = zlut->count;

	return data;
}


int QueuedCUDATracker::GetResultCount()
{
	return FetchResults();
}



// TODO: Let GPU copy frames from frames to GPU 
void QueuedCUDATracker::ScheduleFrame(uchar *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, const LocalizationJob* jobInfo )
{
	uchar* img = (uchar*)imgptr;
	int bpp = sizeof(float);
	if (pdt == QTrkU8) bpp = 1;
	else if (pdt == QTrkU16) bpp = 2;
	for (int i=0;i<numROI;i++){
		ROIPosition pos = positions[i];
		if (pos.x < 0 || pos.y < 0 || pos.x + cfg.width > width || pos.y + cfg.height > height)
			continue;

		uchar *roiptr = &img[pitch * pos.y + pos.x * bpp];
		LocalizationJob job = *jobInfo;
		job.zlutIndex = i;
		ScheduleLocalization(roiptr, pitch, pdt, &job);
	}
}

std::string QueuedCUDATracker::GetProfileReport()
{
	float f = 1.0f/batchesDone;

	return "CUDA tracker report: " + SPrintf("%d batches done of size %d", batchesDone, batchSize ) + "\n" +
		SPrintf("Image copying: %f ms per image\n", time_imageCopy*f) +
		SPrintf("QI: %f ms per image\n", time_QI*f) +
		SPrintf("COM: %f ms per image\n", time_COM*f) +
		SPrintf("Z Computing: %f ms per image\n", time_ZCompute*f);
}


