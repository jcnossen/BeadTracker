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
	- Compute ZLUT profiles
	- Depending on localize flags:
		- copy ZLUT profiles (for ComputeBuildZLUT flag)
		- generate compare profile kernel + compute Z kernel (for ComputeZ flag)
	- Unbind image
	- Async copy results to host


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
#include "gpu_utils.h"
#include "ImageSampler.h"

#define LSQFIT_FUNC __device__ __host__
#include "LsqQuadraticFit.h"

#include "Kernels.h"


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

static std::vector<int> cudaDeviceList; 
void SetCUDADevices(int* dev, int numdev) {
	cudaDeviceList.assign(dev,dev+numdev);
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
		for (uint i=0;i<cudaDeviceList.size();i++)
			devices.push_back(new Device(cudaDeviceList[i]));
	} else {
		devices.push_back (new Device(cfg.cuda_device));
	}
	dbgprintf("Using devices: ");
	for (uint i=0;i<devices.size();i++) {
		cudaDeviceProp p; 
		cudaGetDeviceProperties(&p, devices[i]->index);
		dbgprintf("%s%s", p.name, i<devices.size()-1?", ":"\n");
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

	qi_FFT_length = cfg->qi_radialsteps*2;

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

	for (uint i=0;i<devices.size();i++) {
		Device* d = devices[i];
		cudaSetDevice(d->index);
		d->d_qiradialgrid=qi_radialgrid;
		d->d_zlutradialgrid = zlut_radialgrid;
	}
	kernelParams.zlut.img = cudaImageListf::emptyList();
	
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
	streams[0]->OutputMemoryUse();

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


void QueuedCUDATracker::Stream::OutputMemoryUse()
{
	int deviceMem = d_com.memsize() + d_zlutmapping.memsize() + d_QIprofiles.memsize() + d_QIprofiles_reverse.memsize() + d_radialprofiles.memsize() + d_imgmeans.memsize() +
		d_quadrants.memsize() + d_resultpos.memsize() + d_zlutcmpscores.memsize() + images.totalNumBytes();

	int hostMem = hostImageBuf.memsize() + com.memsize() + zlutmapping.memsize() + results.memsize();

	dbgprintf("Stream memory use: %d kb pinned on host, %d kb device memory (%d for images). \n", hostMem / 1024, deviceMem/1024, images.totalNumBytes()/1024);
}


QueuedCUDATracker::Stream* QueuedCUDATracker::CreateStream(Device* device)
{
	Stream* s = new Stream();

	try {
		s->device = device;
		cudaSetDevice(device->index);
		cudaStreamCreate(&s->stream);

		s->images = cudaImageListf::alloc(cfg.width, cfg.height, batchSize);
		s->images.allocateHostImageBuffer(s->hostImageBuf);

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
		s->d_shiftbuffer.init(qi_FFT_length * batchSize);
		
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
	} catch (...) {
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
		for (uint a=0;a<streams.size();a++) {
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
	return CheckAllStreams(Stream::StreamIdle) && (!currentStream || currentStream->jobs.empty() );
}

bool QueuedCUDATracker::CheckAllStreams(Stream::State s)
{
	FetchResults();
	for (uint a=0;a<streams.size();a++){
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
	LocalizationJob job = *jobInfo;
	job.locType = jobInfo->LocType();
	if (s->device->zlut.isEmpty())  // dont do ZLUT commands when no ZLUT has been set
		job.locType &= ~(LocalizeZ | LocalizeBuildZLUT);
	s->jobs.push_back(job);
	s->localizeFlags |= job.locType; // which kernels to run
	s->zlutmapping[jobIndex].locType = job.LocType();
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
			ExecuteBatch<ImageSampler_Tex> (s);
		else
			ExecuteBatch<ImageSampler_MemCopy> (s);
	}

	s->unlock();
}


void QueuedCUDATracker::Flush()
{
	if (currentStream && currentStream->state == Stream::StreamIdle) {
		currentStream->lock();

		if (useTextureCache) 
			ExecuteBatch<ImageSampler_Tex> (currentStream);
		else 
			ExecuteBatch<ImageSampler_MemCopy> (currentStream);

		currentStream->unlock();
		currentStream = 0;
	}
}


#ifdef QI_DBG_EXPORT
static unsigned long hash(unsigned char *str, int n)
{
    unsigned long hash = 5381;
    
    for (int i=0;i<n;i++) {
		int c = str[i];
        hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
	}

    return hash;
}
#endif

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
	dim3 qdrThreads(16, 8);


	if (0) {
		dim3 qdrDim( (njobs + qdrThreads.x - 1) / qdrThreads.x, (cfg.qi_radialsteps + qdrThreads.y - 1) / qdrThreads.y, 4 );
		QI_ComputeQuadrants<TImageSampler> <<< qdrDim , qdrThreads, 0, s->stream >>> 
			(njobs, s->images, initial->data, s->d_quadrants.data, s->d_imgmeans.data, kernelParams.qi);

		QI_QuadrantsToProfiles <<< blocks(njobs), threads(), 0, s->stream >>> 
			(njobs, s->images, s->d_quadrants.data, s->d_QIprofiles.data, s->d_QIprofiles_reverse.data, kernelParams.qi);
	}
	else {
		QI_ComputeProfile <TImageSampler> <<< blocks(njobs), threads(), 0, s->stream >>> (njobs, s->images, initial->data, 
			s->d_quadrants.data, s->d_QIprofiles.data, s->d_QIprofiles_reverse.data, s->d_imgmeans.data,  kernelParams.qi);
	}
	/*
	cudaStreamSynchronize(s->stream);
	auto q0 = s->d_quadrants.toVector();
	auto p0 = s->d_QIprofiles.toVector();

	WriteImageAsCSV("qi-qtc.txt", &q0[0], cfg.qi_radialsteps * 4, njobs);
	WriteComplexImageAsCSV("qi-ptc.txt", (std::complex<float>*)&p0[0], 2*qi_FFT_length, njobs);

	QI_ComputeProfile <TImageSampler> <<< blocks(njobs), threads(), 0, s->stream >>> (njobs, s->images, initial->data, 
		s->d_quadrants.data, s->d_QIprofiles.data, s->d_QIprofiles_reverse.data, s->d_imgmeans.data,  kernelParams.qi);
	cudaStreamSynchronize(s->stream);
	auto q1 = s->d_quadrants.toVector();
	auto p1 = s->d_QIprofiles.toVector();

	WriteImageAsCSV("qi-q1.txt", &q1[0], cfg.qi_radialsteps * 4, njobs);
	WriteComplexImageAsCSV("qi-p1.txt", (std::complex<float>*) &p1[0], 2*qi_FFT_length, njobs);

	for (int j=0;j<njobs;j++) {
		float2* r1 = &p1[j * cfg.qi_radialsteps * 4];
		float2* r0 = &p0[j * cfg.qi_radialsteps * 4];

		float* s1 = &q1[j * cfg.qi_radialsteps * 4];
		float* s0 = &q0[j * cfg.qi_radialsteps * 4];

		for (int q=0;q<4;q++) {
			for (int r=0;r<cfg.qi_radialsteps;r++) {

				s1 ++;
				s0 ++;
			}
		}


	}
	*/
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
	QI_OffsetPositions<<<blocks(njobs), threads(), 0, s->stream>>>
		(njobs, initial->data, newpos->data, prof, qi_FFT_length, d_offsets, pixelsPerProfLen, s->d_shiftbuffer.data); 
}


template<typename TImageSampler>
void QueuedCUDATracker::ExecuteBatch(Stream *s)
{
	if (s->JobCount()==0)
		return;
	//dbgprintf("Sending %d images to GPU stream %p...\n", s->jobCount, s->stream);

	Device *d = s->device;
	cudaSetDevice(d->index);
	kernelParams.qi.radialgrid = d->d_qiradialgrid.data;
	kernelParams.zlut.img = d->zlut;
	kernelParams.zlut.radialgrid = d->d_zlutradialgrid.data;
	kernelParams.zlut.zcmpwindow = d->zcompareWindow.data;

	cudaEventRecord(s->batchStart, s->stream);

	
	{ProfileBlock p("image to gpu");
	s->images.copyToDevice(s->hostImageBuf.data(), true, s->stream); }
	//cudaMemcpy2DAsync( s->images.data, s->images.pitch, s->hostImageBuf.data(), sizeof(float)*s->images.w, s->images.w*sizeof(float), s->images.h * s->JobCount(), cudaMemcpyHostToDevice, s->stream); }
	//{ ProfileBlock p("jobs to gpu");
	//s->d_jobs.copyToDevice(s->jobs.data(), s->jobCount, true, s->stream); }
	cudaEventRecord(s->imageCopyDone, s->stream);

	TImageSampler::BindTexture(s->images);
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
	TImageSampler::UnbindTexture(s->images);
	cudaEventRecord(s->zcomputeDone, s->stream);

	{ ProfileBlock p("Results to host");
	curpos->copyToHost(s->results.data(), true, s->stream);}

	// Make sure we can query the all done signal
	cudaEventRecord(s->localizationDone, s->stream);

	s->state = Stream::StreamExecuting;
}


int QueuedCUDATracker::FetchResults()
{
	for (uint a=0;a<streams.size();a++) {
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
		if ( !(s->jobs[a].locType & LocalizeZ))
			r.pos.z = 0.0f;

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
	
	for (uint i=0;i<devices.size();i++) {
		devices[i]->SetZLUT(data, cfg.zlut_radialsteps, planes, numLUTs, zcmp);
	}

	for (uint i=0;i<streams.size();i++) {
		StreamUpdateZLUTSize(streams[i]);
	}
}

void QueuedCUDATracker::StreamUpdateZLUTSize(Stream* s)
{		
	cudaSetDevice(s->device->index);
	s->d_zlutcmpscores.init(s->device->zlut.h * batchSize);
}

void QueuedCUDATracker::Device::SetZLUT(float *data, int radialsteps, int planes, int numLUTs, float* zcmp)
{
	cudaSetDevice(index);

	if (zcmp)
		zcompareWindow.copyToDevice(zcmp, radialsteps, false);
	else 
		zcompareWindow.free();

	zlut = cudaImageListf::alloc(radialsteps, planes, numLUTs);
	if (data) {
		for (int i=0;i<numLUTs;i++)
			zlut.copyImageToDevice(i, &data[planes*radialsteps*i]);
	}
	else zlut.clear();
}	

// delete[] memory afterwards
float* QueuedCUDATracker::GetZLUT(int *count, int* planes)
{
	cudaImageListf* zlut = &devices[0]->zlut;

	float* data = new float[zlut->h * cfg.zlut_radialsteps * zlut->count];
	if (zlut->data) {
		//zlut->copyToHost(data, false);
		for (int i=0;i<zlut->count;i++)
			zlut->copyImageToHost(i, &data[cfg.zlut_radialsteps*zlut->h]);
	} else
		std::fill(data, data+(cfg.zlut_radialsteps*zlut->h*zlut->count), 0.0f);

	if (planes) *planes = zlut->h;
	if (count) *count = zlut->count;

	return data;
}


int QueuedCUDATracker::GetResultCount()
{
	return FetchResults();
}


void QueuedCUDATracker::ScheduleFrame(uchar *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, const LocalizationJob* jobInfo)
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
		job.zlutIndex = i + jobInfo->zlutIndex;
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


