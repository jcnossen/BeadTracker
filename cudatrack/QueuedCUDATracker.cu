#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "std_incl.h"

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
template<bool cpuMode> struct shared_mem {};
template<> struct shared_mem<true> {
	CUBOTH static float2* sharedMemory(float2* sharedBuf) { return sharedBuf; }
};
template<> struct shared_mem<false> {
	CUBOTH static float2* sharedMemory(float2* sharedBuf) { return cudaSharedMemory; }
};

// QueuedCUDATracker allows runtime choosing of GPU or CPU code. All GPU kernel calls are done through the following macro:
// Depending on 'useCPU' it either invokes a CUDA kernel named 'Funcname', or simply loops over the data on the CPU side calling 'Funcname' for each image
#define KERNEL_DISPATCH(Funcname, TParam) \
__global__ void Funcname##Kernel(cudaImageListf images, TParam param) { \
	int idx = blockIdx.x * blockDim.x + threadIdx.x; \
	if (idx < images.count) { \
		Funcname<false>(idx, images, param); \
	} \
} \
void QueuedCUDATracker::CallKernel_##Funcname(cudaImageListf& images, TParam param, uint sharedMem)  { \
	if (useCPU) { \
		for (int idx=0;idx<images.count;idx++) { \
			::Funcname <true> (idx, images, param); \
		} \
	} else { \
		Funcname##Kernel <<<blocks(images.count), threads(), sharedMem>>> (images,param); \
	} \
}

QueuedTracker* CreateQueuedTracker(QTrkSettings* cfg)
{
	return new QueuedCUDATracker(cfg);
}

QueuedCUDATracker::QueuedCUDATracker(QTrkSettings *cfg)
{
	this->cfg = *cfg;

	cudaGetDeviceProperties(&deviceProp, 0);

	batchSize = numThreads * deviceProp.multiProcessorCount;

//	forward_fft = new cudafft<float>(cfg->xc1_profileLength, false);
//	backward_fft = new cudafft<float>(cfg->xc1_profileLength, true);

	//int sharedSpacePerThread = (prop.sharedMemPerBlock-forward_fft->kparams_size*2) / numThreads;
//	dbgprintf("2X FFT instance requires %d bytes. Space per thread: %d\n", forward_fft->kparams_size*2, sharedSpacePerThread);
	dbgprintf("Device: %s\n", deviceProp.name);
	dbgprintf("Shared memory space:%d bytes. Per thread: %d\n", deviceProp.sharedMemPerBlock, deviceProp.sharedMemPerBlock/numThreads);
	dbgprintf("# of CUDA processors:%d\n", deviceProp.multiProcessorCount);
	dbgprintf("warp size: %d\n", deviceProp.warpSize);

	useCPU = cfg->numThreads == 0;
	qiProfileLen = 1;
	while (qiProfileLen < cfg->qi_radialsteps) qiProfileLen *= 2;

	fft_twiddles = DeviceMem( sfft::fill_twiddles<float> (qiProfileLen) );
	sharedBuf.init(qiProfileLen*2*batchSize);
}

QueuedCUDATracker::~QueuedCUDATracker()
{
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
	return 0;
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



template<bool cpuMode>
static CUBOTH void BgCorrectedCOM(int idx, cudaImageListf images, float2* d_com)
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

	d_com[idx].x = momentX / (float)sum;
	d_com[idx].y = momentY / (float)sum;
}

KERNEL_DISPATCH(BgCorrectedCOM, float2*);

void QueuedCUDATracker::ComputeBgCorrectedCOM(cudaImageListf& images, float2* d_com)
{
	CallKernel_BgCorrectedCOM(images, d_com);
}


template<bool cpuMode>
static CUBOTH void MakeTestImage(int idx, cudaImageListf& images, float3* d_positions)
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


template<bool cpuMode>
static CUBOTH qivalue_t QI_ComputeOffset(qicomplex_t* profile, const QIParams& params, int idx) {
	int nr = params.radialSteps;

	qicomplex_t* reverse;
	reverse = (qicomplex_t*) &shared_mem<cpuMode>::sharedMemory(params.sharedBuf) [idx * nr*2];

	for(int x=0;x<nr*2;x++)
		reverse[x] = profile[nr*2-1-x];

	sfft::fft_forward(nr*2, profile, params.d_twiddles);
	sfft::fft_forward(nr*2, reverse, params.d_twiddles);

	// multiply with conjugate
	for(int x=0;x<nr*2;x++)
		profile[x] = profile[x] * reverse[x].conjugate();

	sfft::fft_inverse(nr*2, profile, params.d_twiddles);
	// fft_out2 now contains the autoconvolution
	// convert it to float
	qivalue_t* autoconv = (qivalue_t*)malloc(sizeof(qivalue_t)*nr*2);
	for(int x=0;x<nr*2;x++)  {
		autoconv[x] = profile[(x+nr)%(nr*2)].real();
	}

	float maxPos = ComputeMaxInterp<qivalue_t, 5>(autoconv, nr*2);
	free(autoconv);
	return (maxPos - nr) / (3.14159265359f * 0.5f);
}


static CUBOTH void ComputeQuadrantProfile(cudaImageListf& images, int idx, float* dst, const QIParams& params, int quadrant, float2 center)
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

template<bool cpuMode>
static CUBOTH void ComputeQI(int idx, cudaImageListf& images, QIParams params)
{
	int nr=params.radialSteps;
	float2 center = params.d_initial[idx];

	float pixelsPerProfLen = (params.maxRadius-params.minRadius)/params.radialSteps;
	bool boundaryHit = false;

	size_t total_required = sizeof(qivalue_t)*nr*4 + sizeof(qicomplex_t)*nr*2;

	qivalue_t* buf = (qivalue_t*)malloc(total_required);
	qivalue_t* q0=buf, *q1=buf+nr, *q2=buf+nr*2, *q3=buf+nr*3;

	qicomplex_t* concat0 = (qicomplex_t*)(buf + nr*4);
	qicomplex_t* concat1 = concat0 + nr;

	for (int k=0;k<params.iterations;k++){
		// check bounds
		boundaryHit = images.boundaryHit(center, params.maxRadius);

		for (int q=0;q<4;q++) {
			ComputeQuadrantProfile(images, idx, buf+q*nr, params, q, center);
		}
		
		// Build Ix = qL(-r) || qR(r)
		// qL = q1 + q2   (concat0)
		// qR = q0 + q3   (concat1)
		for(int r=0;r<nr;r++) {
			concat0[nr-r-1] = qicomplex_t(q1[r]+q2[r]);
			concat1[r] = qicomplex_t(q0[r]+q3[r]);
		}

		float offsetX = QI_ComputeOffset<cpuMode>(concat0, params, idx);

		// Build Iy = qB(-r) || qT(r)
		// qT = q0 + q1
		// qB = q2 + q3
		for(int r=0;r<nr;r++) {
			concat0[r] = qicomplex_t(q0[r]+q1[r]);
			concat1[nr-r-1] = qicomplex_t(q2[r]+q3[r]);
		}
		float offsetY = QI_ComputeOffset<cpuMode>(concat0, params, idx);

		//printf("[%d] OffsetX: %f, OffsetY: %f\n", k, offsetX, offsetY);
		center.x += offsetX * pixelsPerProfLen;
		center.y += offsetY * pixelsPerProfLen;
	}

	params.d_output[idx] = center;
	if (params.d_boundaryHits) params.d_boundaryHits[idx] = boundaryHit;

	free(buf);
}

KERNEL_DISPATCH(ComputeQI, QIParams);


void QueuedCUDATracker::ComputeQI(cudaImageListf& images, float2* d_initial, float2* d_result)
{
	QIParams params;
	params.angularSteps = cfg.qi_angularsteps;
	params.d_boundaryHits = 0;
	params.d_initial = d_initial;
	params.d_output = d_result;
	params.iterations = cfg.qi_iterations;
	params.maxRadius = cfg.qi_maxradius;
	params.minRadius = cfg.qi_minradius;
	params.radialSteps = cfg.qi_radialsteps;
	params.d_twiddles = fft_twiddles.data;
	params.sharedBuf = sharedBuf.data;
	params.useShared = cfg.qi_radialsteps*2 * images.count < deviceProp.sharedMemPerBlock;

	uint sharedMemSize = params.useShared ? sizeof(float2) * cfg.qi_radialsteps*2*images.count : 0; 

	if (!useCPU) {
		images.bind(qi_image_texture);
	}

	CallKernel_ComputeQI(images, params, sharedMemSize);

	if (!useCPU)
		images.unbind(qi_image_texture);
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
	b->imageBuf = cudaImageListf::alloc(cfg.width,cfg.height,batchSize, useCPU);
	return b;
}

