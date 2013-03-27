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

template<typename T>
struct cudaImageList;
typedef cudaImageList<float> cudaImageListf;

struct QIParams {
	float minRadius, maxRadius;
	int radialSteps, iterations, angularSteps;
	float2* radialgrid; // precomputed radial directions (cos,sin pairs)
};

struct ZLUTParams {
	CUBOTH float* GetZLUT(int bead, int plane) { return img.pixelAddress(0, plane, bead); }
	float minRadius, maxRadius;
	float* zcmpwindow;
	int angularSteps;
	int planes;
	cudaImageListf img;
	CUBOTH int radialSteps() { return img.w; }
	float2* radialgrid; // precomputed radial directions (cos,sin pairs)
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


struct CUDATrackerJob {
	CUDATrackerJob () { 
		locType=LocalizeOnlyCOM; id=0; zlut=0; 
		initialPos.x=initialPos.y=initialPos.z=0.0f; 
		error=0; firstGuess.x=firstGuess.y=0.0f;
	}

	uint id;
	uint zlut;
	float3 initialPos;
	uint zlutPlane;
	LocalizeType locType;
	float3 resultPos;
	float2 firstGuess;
	uint error;
};


class QueuedCUDATracker : public QueuedTracker {
public:
	QueuedCUDATracker(QTrkSettings* cfg, int batchSize=-1, bool debugStream=false);
	~QueuedCUDATracker();

	bool ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane) override;
	
	// Schedule an entire frame at once, allowing for further optimizations
	void ScheduleFrame(uchar *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, 
		LocalizeType locType, uint frame, uint zlutPlane, bool async) override;
	void WaitForScheduleFrame(uchar* imgptr) override; // Wait for an asynchronous call to ScheduleFrame to be finished with the specified buffer

	void ClearResults() override;

	// data can be zero to allocate ZLUT data.
	void SetZLUT(float* data,  int numLUTs, int planes, float* zcmp=0) override; 
	float* GetZLUT(int *count=0, int* planes=0) override; // delete[] memory afterwards
	int PollFinished(LocalizationResult* results, int maxResults) override;

	// Force the current waiting batch to be processed. Useful when number of localizations is not a multiple of internal batch size (almost always)
	void Flush() override;
	
	bool IsQueueFilled() override;
	bool IsIdle() override;
	int GetResultCount() override;

protected:

	struct Stream {
		Stream();
		~Stream();
		bool IsExecutionDone();
		int CalcMemoryUse();
		int GetJobCount();
		
		pinned_array<float3> results;
		pinned_array<float3> com;
		pinned_array<CUDATrackerJob> jobs;
		device_vec<CUDATrackerJob> d_jobs;
		int jobCount;
		
		cudaImageListf images; 
		//pinned_array<float, cudaHostAllocWriteCombined> hostImageBuf; // original image format pixel buffer
		pinned_array<float> hostImageBuf; // original image format pixel buffer

		// CUDA objects
		cudaStream_t stream; // Stream used
		cufftHandle fftPlan; // a CUFFT plan can be used for both forward and inverse transforms
		cudaEvent_t localizationDone;

		// Intermediate data
		device_vec<float3> d_resultpos;
		device_vec<float3> d_com; // z is zero
		device_vec<float2> d_QIprofiles;
		device_vec<float2> d_QIprofiles_reverse;
		device_vec<float> d_quadrants;

		device_vec<float> d_radialprofiles;// [ radialsteps * njobs ] for Z computation
		device_vec<float> d_zlutcmpscores; // [ zlutplanes * njobs ]
		device_vec<float> d_imgmeans; // image mean value [njobs]

		uint localizeFlags; // Indicates whether kernels should be ran for building zlut, z computing, or QI

		Threads::Mutex mutex; // Mutex to lock when queing jobs or copying results
		void lock() { mutex.lock(); }
		void unlock() { mutex.unlock(); }

		enum State {
			StreamIdle,
			StreamExecuting
		};
		volatile State state; // I'm assuming this variable is atomic
	};
	Stream* CreateStream();
	void CopyStreamResults(Stream* s);

	bool debugStream;
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
	
	// QI profiles need to have power-of-two dimensions. qiProfileLen stores the closest power-of-two value that is bigger than cfg.qi_radialsteps
	int qi_FFT_length ;
	cudaDeviceProp deviceProp;
	KernelParams kernelParams;

	int zlut_count;
	cudaImageListf zlut;
	device_vec<float> zcompareWindow;
	device_vec<float2> d_qiradialgrid;
	device_vec<float2> d_zlutradialgrid;

	int FetchResults();
	void ExecuteBatch(Stream *s);
	Stream* GetReadyStream(); // get a stream that not currently executing, and still has room for images
	void QI_Iterate(device_vec<float3>* initial, device_vec<float3>* newpos, Stream *s);
	bool CheckAllStreams(Stream::State state);

public:
	typedef std::map<const char*, std::pair<int, double> > ProfileResults;
	static ProfileResults GetProfilingResults();
};


