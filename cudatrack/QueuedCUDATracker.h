// CUDA implementation of QueuedTracker interface

// Thread-Safety:

// We assume 2 threads concurrently accessing the tracker functions:
//	- Queueing thread: ScheduleLocalization, SetZLUT, GetZLUT, Flush, IsQueueFilled, IsIdle
//	- Fetching thread: PollFinished, GetResultCount, ClearResults

#pragma once
#include "QueuedTracker.h"
#include "threads.h"
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <list>
#include <vector>
#include <map>
#include "gpu_utils.h"
#include "cudaImageList.h"

template<typename T>
struct cudaImageList;
typedef cudaImageList<float> cudaImageListf;

struct QIParams {
	float minRadius, maxRadius;
	int radialSteps, iterations, trigtablesize, angularSteps;
	float2* cos_sin_table;
};

struct ZLUTParams {
	CUBOTH float* GetZLUT(int bead, int plane) { return img.pixelAddress(0, plane, bead); }
	float minRadius, maxRadius;
	float* zcmpwindow;
	int angularSteps;
	int planes;
	cudaImageListf img;
	CUBOTH int radialSteps() { return img.w; }
	float2* trigtable; // precomputed radial directions (cos,sin pairs)
};

struct KernelParams {
	QIParams qi;
	ZLUTParams zlut;
	int sharedMemPerThread;
	float com_bgcorrection;
};


struct QIParamWrapper {
	KernelParams kernel;
	float2* d_initial;
	float2* d_result;
};

struct ZLUTMapping {
	int zlutIndex, zlutPlane; // if <0 then, no zlut for this job
	LocalizeType locType;
};

class QueuedCUDATracker : public QueuedTracker {
public:
	QueuedCUDATracker(const QTrkComputedConfig& cc, int batchSize=-1);
	~QueuedCUDATracker();
	void EnableTextureCache(bool useTextureCache) { this->useTextureCache=useTextureCache; }
	
	void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo) override;
	
	// Schedule an entire frame at once, allowing for further optimizations
	void ScheduleFrame(uchar *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo) override;

	void ClearResults() override;

	// data can be zero to allocate ZLUT data.
	void SetZLUT(float* data,  int numLUTs, int planes, float* zcmp=0) override; 
	float* GetZLUT(int *count=0, int* planes=0) override; // delete[] memory afterwards
	int PollFinished(LocalizationResult* results, int maxResults) override;

	std::string GetProfileReport() override;

	// Force the current waiting batch to be processed. Useful when number of localizations is not a multiple of internal batch size (almost always)
	void Flush() override;
	
	int GetQueueLength(int* maxQueueLen) override;
	bool IsIdle() override;
	int GetResultCount() override;

protected:

	struct Device {
		Device(int index) {this->index=index; zlut=cudaImageListf::emptyList(); }
		~Device(); 
		void SetZLUT(float *data, int radialsteps, int planes, int numLUTs, float* zcmp);

		cudaImageListf zlut;
		device_vec<float> zcompareWindow;
		device_vec<float2> d_qi_trigtable;
		device_vec<float2> d_zlut_trigtable;
		int index;
	};

	struct Stream {
		Stream();
		~Stream();
		bool IsExecutionDone();
		void OutputMemoryUse();
		int JobCount() { return jobs.size(); }
		
		pinned_array<float3> results;
		pinned_array<float3> com;
		pinned_array<ZLUTMapping> zlutmapping;
		device_vec<ZLUTMapping> d_zlutmapping;
		std::vector<LocalizationJob> jobs;
		
		cudaImageListf images; 
		pinned_array<float, cudaHostAllocWriteCombined> hostImageBuf; // original image format pixel buffer
		//pinned_array<float> hostImageBuf; // original image format pixel buffer

		// CUDA objects
		cudaStream_t stream; // Stream used
		cufftHandle fftPlan; // a CUFFT plan can be used for both forward and inverse transforms

		// Events
		cudaEvent_t localizationDone; // all done.
		// Events for profiling
		cudaEvent_t imageCopyDone, comDone, qiDone, zcomputeDone, batchStart;

		// Intermediate data
		device_vec<float3> d_resultpos;
		device_vec<float3> d_com; // z is zero
		device_vec<float2> d_QIprofiles;
		device_vec<float2> d_QIprofiles_reverse;
		device_vec<float> d_quadrants;
		
		device_vec<float> d_radialprofiles;// [ radialsteps * njobs ] for Z computation
		device_vec<float> d_zlutcmpscores; // [ zlutplanes * njobs ]
		device_vec<float> d_imgmeans; // image mean value [njobs]

		device_vec<float> d_shiftbuffer; // [QI_fftlength * njobs] ComputeMaxInterp temp space

		uint localizeFlags; // Indicates whether kernels should be ran for building zlut, z computing, or QI
		Device* device;

		Threads::Mutex mutex; // Mutex to lock when queing jobs or copying results
		void lock() { mutex.lock(); }
		void unlock() { mutex.unlock(); }

		enum State {
			StreamIdle,
			StreamExecuting
		};
		volatile State state; // I'm assuming this variable is atomic
	};

	int numThreads;
	int batchSize;

	dim3 blocks(int workItems) {
		return dim3((workItems+numThreads-1)/numThreads);
	}
	dim3 blocks() {
		return dim3((batchSize+numThreads-1)/numThreads);
	}
	dim3 threads() {
		return dim3(numThreads);
	}

	std::vector<Stream*> streams;
	Stream* currentStream;
	std::list<LocalizationResult> results;
	std::vector<Device*> devices;
	bool useTextureCache; // speed up, but more numerical errors
	
	// QI profiles need to have power-of-two dimensions. qiProfileLen stores the closest power-of-two value that is bigger than cfg.qi_radialsteps
	int qi_FFT_length;
	cudaDeviceProp deviceProp;
	KernelParams kernelParams;

	int FetchResults();
	template<typename TImageSampler> void ExecuteBatch(Stream *s);
	Stream* GetReadyStream(); // get a stream that not currently executing, and still has room for images
	template<typename TImageSampler> void QI_Iterate(device_vec<float3>* initial, device_vec<float3>* newpos, Stream *s, int angularSteps);
	bool CheckAllStreams(Stream::State state);
	void InitializeDeviceList();
	Stream* CreateStream(Device* device);
	void CopyStreamResults(Stream* s);
	void StreamUpdateZLUTSize(Stream *s);

public:
	typedef std::map<const char*, std::pair<int, double> > ProfileResults;
	static ProfileResults GetProfilingResults();

	// Profiling
	double time_COM, time_QI, time_imageCopy, time_ZCompute;
	int batchesDone;
	std::string deviceReport;
};


