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

#define LSQFIT_FUNC __device__ __host__
#include "LsqQuadraticFit.h"

// Types used by QI algorithm
typedef float qivalue_t;
typedef sfft::complex<qivalue_t> qicomplex_t;


// According to this, textures bindings can be switched after the asynchronous kernel is launched
// https://devtalk.nvidia.com/default/topic/392245/texture-binding-and-stream/
texture<float, cudaTextureType2D, cudaReadModeElementType> qi_image_texture(0,  cudaFilterModeLinear); // Un-normalized


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
	while (batchSize * cfg->height > deviceProp.maxTexture2D[1]) {
		batchSize/=2;
	}
	this->batchSize = batchSize;

	//int sharedSpacePerThread = (prop.sharedMemPerBlock-forward_fft->kparams_size*2) / numThreads;
//	dbgprintf("2X FFT instance requires %d bytes. Space per thread: %d\n", forward_fft->kparams_size*2, sharedSpacePerThread);
	dbgprintf("Device: %s\n", deviceProp.name);
	dbgprintf("Shared memory space:%d bytes. Per thread: %d\n", deviceProp.sharedMemPerBlock, deviceProp.sharedMemPerBlock/numThreads);
	dbgprintf("# of CUDA processors:%d\n", deviceProp.multiProcessorCount);
	dbgprintf("Warp size: %d. Batch size: %d\n", deviceProp.warpSize, batchSize);

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
	std::vector<float2> qiradialgrid(qi.angularSteps);
	for (int i=0;i<qi.angularSteps;i++)  {
		float ang = 0.5f*3.141593f*i/(float)qi.angularSteps;
		qiradialgrid[i]=make_float2(cos(ang), sin(ang));
	}
	d_qiradialgrid=qiradialgrid;
	qi.radialgrid=d_qiradialgrid.data;
	
	zlut = cudaImageListf::empty();
	kernelParams.zlut.img = zlut;

//	results.reserve(50000);
	
	streams.resize(cfg->numThreads);
	for (int i=0;i<streams.size();i++) {
		streams[i] = CreateStream();
	}
	currentStream=streams[0];
	int memUsePerStream = streams[0]->CalcMemoryUse();
	dbgprintf("Stream memory use: %d kb", memUsePerStream/1024);
}

QueuedCUDATracker::~QueuedCUDATracker()
{
	if (zlut.data)
		zlut.free();
	
	DeleteAllElems(streams);
}

__global__ void TestSampleImage(cudaImageListf images, int idx, float* dst)
{
	int x = threadIdx.x + blockDim.x * blockIdx.x;
	int y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x < images.w && y < images.h) {

		float p = tex2D( qi_image_texture, x, y);
		//float& p = images.pixel(x, y, idx);
		dst [y*images.w+x] = p;
	}
}

void TestCopyImage( cudaImageListf& images, int idx, const char* file)
{
	device_vec<float> ddst (images.w*images.h);
	int nt=32;
	cudaDeviceSynchronize();
	TestSampleImage <<< dim3((images.w+nt-1)/nt, (images.h+nt-1)/nt), dim3(nt,nt) >>> (images, idx, ddst.data);

	std::vector<float> dst = ddst;
	uchar* norm = floatToNormalizedInt(&dst[0], images.w,images.h, (uchar)255);
	WriteJPEGFile(norm, images.w,images.h, file, 99);
	delete[] norm;
}


static __device__ float2 BgCorrectedCOM(int idx, cudaImageListf images, float correctionFactor)
{
	int imgsize = images.w*images.h;
	float sum=0, sum2=0;
	float momentX=0;
	float momentY=0;

	for (int y=0;y<images.h;y++)
		for (int x=0;x<images.w;x++) {
			float v = tex2D(qi_image_texture, x, y + idx*images.h);
			//float v = images.pixel(x,y,idx);
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
			float v = tex2D(qi_image_texture, x, y + idx*images.h);
			//float v = images.pixel(x,y,idx);
			v = fabsf(v-mean)-correctionFactor*stdev;
			if(v<0.0f) v=0.0f;
			sum += v;
			momentX += (x+0.5f)*v;
			momentY += (y+0.5f)*v;
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

int QueuedCUDATracker::Stream::CalcMemoryUse()
{
	return d_com.memsize() + d_jobs.memsize() + d_QIprofiles.memsize() + d_quadrants.memsize() + d_resultpos.memsize();
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
	s->d_quadrants.init(qi_FFT_length*batchSize*2);
	s->d_QIprofiles.init(batchSize*4*qi_FFT_length); // (2 axis) * (2 radialsteps) * (rev/forw) = 8 * nr = 4 * qi_FFT_length
	s->d_QIprofiles_reverse = s->d_QIprofiles.data + batchSize*2*qi_FFT_length;
	
	// 2* batchSize, since X & Y both need an FFT transform
	//cufftResult_t r = cufftPlanMany(&s->fftPlan, 1, &qi_FFT_length, 0, 1, qi_FFT_length, 0, 1, qi_FFT_length, CUFFT_C2C, batchSize*4);
	cufftResult_t r = cufftPlan1d(&s->fftPlan, qi_FFT_length, CUFFT_C2C, batchSize*2);

	if(r != CUFFT_SUCCESS) {
		throw std::runtime_error( SPrintf("CUFFT plan creation failed. FFT len: %d. Batchsize: %d\n", qi_FFT_length, batchSize*4));
	}
	cufftSetCompatibilityMode(s->fftPlan, CUFFT_COMPATIBILITY_NATIVE);
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
		Threads::Sleep(1);
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
	CopyImageToFloat(data, cfg.width, cfg.height, pitch, pdt, hostbuf);

//	tmp = floatToNormalizedInt( (float*)hostbuf, cfg.width,cfg.height,(uchar)255);
//	WriteJPEGFile(tmp, cfg.width,cfg.height, "writehostbuf2.jpg", 99);
//	delete[] tmp;

	// If batch is filled, copy the image to video memory asynchronously, and start the localization
	if (s->jobCount == batchSize)
		ExecuteBatch(s);

	s->unlock();

	return true;
}

/*
		q1: xprof[len-r-1], yprof[r]
		q2: xprof[len-r-1], yprof[len-r-1]
		q3: xprof[r], yprof[len-r-1]
		q0: xprof[r], yprof[r]
*/
static __device__ void QI_ComputeProfile2(cudaImageListf& images, int idx, float* dst, const QIParams& params, int quadrant, float2 center)
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
			float x = center.x + mx*params.radialgrid[a].x * r;
			float y = center.y + my*params.radialgrid[a].y * r;
			sum += images.interpolate(x, y, idx);
		}
		dst[i] = sum/params.angularSteps-images.borderValue;
		total += dst[i];
	}
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
	
	float sum2=0.0f;
	float total = 0.0f;
	float rstep = (params.maxRadius - params.minRadius) / params.radialSteps;
	for (int i=0;i<params.radialSteps; i++) {
		float sum = 0.0f;
		float r = params.minRadius + rstep * i;

		for (int a=0;a<params.angularSteps;a++) {
			//float ang = 0.5f*3.141593f*i/(float)params.angularSteps;
	//		float x = center.x + mx*cosf(ang) * r;
//			float y = center.y + my*sinf(ang) * r;
			float x = center.x + mx*params.radialgrid[a].x * r;
			float y = center.y + my*params.radialgrid[a].y * r;
			//float v = images.interpolate(x, y, idx);;
			//sum += v;
			sum += tex2D(qi_image_texture, x,y + idx*images.h);
//			printf("[%d] sum[%d,%d]:%f\n", idx, a, i, v);
	//		printf("%f; ", v);
		}
		//printf("\n");

		dst[i] = sum/params.angularSteps-images.borderValue;
		total += dst[i];
		sum2+=sum;
	}

//	printf("[%d] sum2:%f\n", idx,sum2);

}

__global__ void QI_ComputeProfile(int count, cudaImageListf images, float3* initial,float3* dstpos, float* quadrants, float2* profiles, float2* reverseProfiles, QIParams params)
{
//ComputeQuadrantProfile(cudaImageListf& images, int idx, float* dst, const QIParams& params, int quadrant, float2 center)
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx < count) {
		int fftlen = params.radialSteps*2;
		float* img_qdr = &quadrants[ idx * params.radialSteps * 4 ];
		for (int q=0;q<4;q++)
			ComputeQuadrantProfile(images, idx, &img_qdr[q*params.radialSteps], params, q, make_float2(initial[idx].x, initial[idx].y));

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
	float* shifted = &compute_offset_buffer [threadIdx.x * fftlen];
	int nr = fftlen/2;
	for(int x=0;x<fftlen;x++)  {
		shifted[x] = autoconv[(x+nr)%(nr*2)].x;
	}

	float maxPos = ComputeMaxInterp<float,7>(shifted, fftlen);
	float offset = (maxPos - nr) / (3.14159265359f * 0.5f);
	return offset;
}

__global__ void QI_OffsetPositions(int njobs, float3* current, float3* dst, cufftComplex* autoconv, int fftLength, float2* offsets)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < njobs) {
		// X
		cufftComplex* autoconvX = &autoconv[idx * fftLength * 2];
		float xoffset = QI_ComputeAxisOffset(autoconvX, fftLength);

		cufftComplex* autoconvY = autoconvX + fftLength;
		float yoffset = QI_ComputeAxisOffset(autoconvY, fftLength);

		dst[idx].x = current[idx].x + xoffset;
		dst[idx].y = current[idx].y + yoffset;

		if (offsets) 
			offsets[idx] = make_float2( xoffset, yoffset);
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
#ifdef _DEBUG
	uchar* cp = (uchar*)ALLOCA(elemsize*numelem*sizeof(T));
	cudaDeviceSynchronize();
	cudaMemcpy(cp, data, sizeof(T)*elemsize*numelem, cudaMemcpyDeviceToHost);

	dbgprintf("%s:\n", name);
	for (int i=0;i<numelem;i++) {
		uchar *elem = cp+elemsize*sizeof(T)*i;
		dbgprintf("[%d]: %d\n", i, hash(elem, elemsize));
		for (int j=0;j<elemsize/4;j++) {
//			float* d = (float*)elem;

		}
	}
#endif
}

void QueuedCUDATracker::QI_Iterate(device_vec<float3>* initial, device_vec<float3>* newpos, Stream *s)
{
/*	int njobs = s->jobs.size();
	int nElem = njobs * qi_FFT_length * 2; // 2 profiles of qi_FFT_length size for each job
	QI_ComputeProfile <<< blocks(nElem), threads(), 0, s->stream >>> (qi_FFT_length, s->*/

//	cudaDeviceSynchronize();
//	std::vector<float3> initial_h=*initial;

//	TestCopyImage(s->images, 0, "testimg.jpg");

	int njobs = s->jobCount;
	QI_ComputeProfile <<< blocks(njobs), threads(), 0, s->stream >>> (njobs, s->images, initial->data, newpos->data, 
		s->d_quadrants.data, s->d_QIprofiles.data, s->d_QIprofiles_reverse, kernelParams.qi);

	checksum(s->d_quadrants.data, qi_FFT_length * 2, njobs, "quadrant");
	checksum(s->d_QIprofiles.data, qi_FFT_length * 2, njobs, "prof");
	checksum(s->d_QIprofiles_reverse, qi_FFT_length * 2, njobs, "revprof");

#ifdef QI_DBG_EXPORT
	cudaDeviceSynchronize();
	std::vector<float> hquadrants = s->d_quadrants;
	WriteImageAsCSV("quadrants.txt", (float*)&hquadrants[0], qi_FFT_length*2, njobs);
#endif

	cufftComplex* prof = (cufftComplex*)s->d_QIprofiles.data;
	cufftComplex* revprof = s->d_QIprofiles_reverse;
#ifdef QI_DBG_EXPORT
	cudaDeviceSynchronize();
	std::vector<float2> hprof = s->d_QIprofiles;
	WriteComplexImageAsCSV("profiles.txt", (std::complex<float>*)&hprof[0], qi_FFT_length*2, njobs*2);
#endif

	cufftExecC2C(s->fftPlan, prof, prof, CUFFT_FORWARD);
	cufftExecC2C(s->fftPlan, revprof, revprof, CUFFT_FORWARD);
#ifdef QI_DBG_EXPORT
	cudaDeviceSynchronize();
	std::vector<float2> hprof_fft = s->d_QIprofiles;
	WriteComplexImageAsCSV("fftprofiles.txt", (std::complex<float>*)&hprof_fft[0], qi_FFT_length*2, njobs);
#endif

	int nval = qi_FFT_length * 2 * batchSize, nthread=256;
	QI_MultiplyWithConjugate<<< dim3( (nval + nthread - 1)/nthread ), dim3(nthread), 0, s->stream >>>(nval, prof, revprof);
	cufftExecC2C(s->fftPlan, prof, prof, CUFFT_INVERSE);

	QI_OffsetPositions<<<blocks(njobs), threads(), sizeof(float)*qi_FFT_length*numThreads , s->stream>>>(njobs, initial->data, newpos->data, prof, qi_FFT_length, 0);
#ifdef QI_DBG_EXPORT
	device_vec<float2> offsets(njobs);
	QI_OffsetPositions<<<blocks(njobs), threads(), sizeof(float)*qi_FFT_length*numThreads, s->stream>>>(njobs, initial->data, newpos->data, prof, qi_FFT_length, offsets.data);
	cudaDeviceSynchronize();
	std::vector<float2> h_offsets = offsets;
	for (int i=0;i<njobs;i++) {
		dbgprintf("Offset[%d]: x: %f, y: %f\n", i, h_offsets[i].x, h_offsets[i].y);
	}
	std::vector<float2> qiprof = s->d_QIprofiles;
	WriteComplexImageAsCSV("autoconv.txt", (std::complex<float>*)&qiprof[0], qi_FFT_length*2, njobs);
#endif
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

	cudaMemcpy2DAsync( s->images.data, s->images.pitch, s->hostImageBuf.data(), sizeof(float)*s->images.w, s->images.w*sizeof(float), s->images.h * s->jobCount, cudaMemcpyHostToDevice, s->stream);
	s->d_jobs.copyToDevice(s->jobs.data(), s->jobCount, true, s->stream);
	s->images.bind(qi_image_texture);
	BgCorrectedCOM <<< blocks(s->jobCount), threads(), 0, s->stream >>> (s->jobCount, s->images, s->d_com.data, cfg.com_bgcorrection);

	checksum(s->d_com.data, 2, s->jobCount, "com");

	device_vec<float3> *curpos = &s->d_com;
	for (int a=0;a<cfg.qi_iterations;a++) {
		QI_Iterate(curpos, &s->d_resultpos, s);
		curpos = &s->d_resultpos;
	}

	if (s->localizeFlags & LocalizeBuildZLUT) {
		BuildZLUTKernel <<< blocks(s->jobs.size()), threads(), 0, s->stream >>> (s->jobCount, s->images, kernelParams.zlut, curpos->data, s->d_jobs.data);
	//	TestCopyImage( s->images, 0, "qtrktestimg0.jpg");
	}
	
	curpos->copyToHost(s->results.data(), true, s->stream);
	
	s->images.unbind(qi_image_texture);
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
		dstResults[numResults++] = results.front();
		results.pop_front();
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
