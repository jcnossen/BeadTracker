#pragma once
#include "threads.h"
#include "QueuedTracker.h"
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <list>
#include <vector>
#include "gpu_utils.h"

template<typename T>
struct cudaImageList;
typedef cudaImageList<float> cudaImageListf;

struct QIParams {
	float minRadius, maxRadius;
	int radialSteps, iterations, angularSteps;
};

struct ZLUTParams {
	CUBOTH float* GetZLUT(int bead, int plane) { return &img.pixel(0, plane, bead); }
	float minRadius, maxRadius;
	float* zcmpwindow;
	int angularSteps;
	int planes;
	cudaImageListf img;
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
	QueuedCUDATracker(QTrkSettings* cfg, int batchSize=-1);
	~QueuedCUDATracker();

// Thread-Safety:

// We assume 2 threads concurrently accessing the tracker functions:
//	- Queueing thread: ScheduleLocalization, SetZLUT, 
//	- 

	void Start();
	bool ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane);
	
	// Schedule an entire frame at once, allowing for further optimizations
	void ScheduleFrame(uchar *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, 
		LocalizeType locType, uint frame, uint zlutPlane, bool async);
	void WaitForScheduleFrame(uchar* imgptr); // Wait for an asynchronous call to ScheduleFrame to be finished with the specified buffer

	void ClearResults();

	// data can be zero to allocate ZLUT data.
	void SetZLUT(float* data,  int numLUTs, int planes, int res, float* zcmp=0); 
	float* GetZLUT(int *count=0, int* planes=0, int *res=0); // delete[] memory afterwards
	int PollFinished(LocalizationResult* results, int maxResults);
	void GenerateTestImage(float* dst, float xp,float yp, float z, float photoncount);

	// Force the current waiting batch to be processed. Useful when number of localizations is not a multiple of internal batch size (almost always)
	void Flush();
	
	bool IsQueueFilled();
	bool IsIdle();
	int GetResultCount();

	// Direct kernel wrappers
	void GenerateImages(cudaImageListf& imgList, float3 *d_positions);
	void ComputeBgCorrectedCOM(cudaImageListf& imgList, float2* d_com);
	void Compute1DXCor(cudaImageListf& images, float2* d_initial, float2* d_result);
	void ComputeQI(cudaImageListf& images, float2* d_initial, float2* d_result);
	void Compute1DXCorProfiles(cudaImageListf& images, float* d_profiles);

protected:

	struct Stream {
		Stream();
		~Stream();
		
		pinned_array<uint> localizationFlags; // tells the kernel what to do, per image in the batch
		pinned_array<float3> results;
		std::vector<CUDATrackerJob> jobs;
		
		int jobCount() { return jobs.size(); } 
		cudaImageListf images; 
		pinned_array<float, cudaHostAllocWriteCombined> hostImageBuf; // original image format pixel buffer

		// CUDA objects
		cudaStream_t stream; // Stream used
		cufftHandle fftPlan; // a CUFFT plan can be used for both forward and inverse transforms
		cudaEvent_t localizationDone, imageBufferCopied;

		// Intermediate data
		device_vec<float2> d_com, d_qi;
		device_vec<float2> d_QIprofiles;

		uint localizeFlags; // Indicates whether kernels should be ran for building zlut, z computing, or QI

		enum State {
			StreamIdle,
			StreamExecuting,
			StreamStoringResults
		};
	};
	Stream* CreateStream();
	void CopyBatchResults(Stream* s);

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
	std::vector<LocalizationResult> results;
	
	// QI profiles need to have power-of-two dimensions. qiProfileLen stores the closest power-of-two value that is bigger than cfg.qi_radialsteps
	int qiProfileLen;
	cudaDeviceProp deviceProp;
	KernelParams kernelParams;

	int zlut_count, zlut_planes, zlut_res;
	cudaImageListf zlut;
	device_vec<float> zcompareWindow;

	int FetchResults();
	void ExecuteBatch(Stream *s);
	Stream* GetReadyStream(); // get a stream that not currently executing, and still has room for images
};



