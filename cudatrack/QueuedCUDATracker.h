#pragma once
#include "threads.h"
#include "QueuedTracker.h"
#include <cuda_runtime_api.h>
//#include "cudafft/cudafft.h"
#include <list>
#include <vector>
#include "gpu_utils.h"

template<typename T>
struct cudaImageList;
typedef cudaImageList<float> cudaImageListf;

struct QIParams {
	sfft::complex<float>* d_twiddles;
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
	float2* sharedBuf, *buffer;
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
		locType=LocalizeXCor1D; id=0; zlut=0; 
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

	void Start();
	bool ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane);
	void BatchSchedule(uchar *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, 
									LocalizeType locType, uint frame, uint zlutPlane);
	void ClearResults();

	// data can be zero to allocate ZLUT data.
	void SetZLUT(float* data,  int numLUTs, int planes, int res, float* zcmp=0); 
	float* GetZLUT(int *count=0, int* planes=0, int *res=0); // delete[] memory afterwards
	int PollFinished(LocalizationResult* results, int maxResults);
	void GenerateTestImage(float* dst, float xp,float yp, float z, float photoncount);

	// Force the current waiting batch to be processed. Useful when number of localizations is not a multiple of internal batch size (almost always)
	void Flush();

	// Debug stuff
	float* GetDebugImage() { return 0; }

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

	struct Batch{
		Batch() { hostImageBuf = 0; images.data=0; jobCount=0; }
		~Batch();
		
		pinned_array<CUDATrackerJob> jobs;
		int jobCount; // nr of jobs currently stored in jobs array
		cudaImageListf images; 
		float* hostImageBuf; // original image format pixel buffer, pinned memory with write-combined flags for fast host->device transfer
		device_vec<CUDATrackerJob> d_jobs;
		cudaEvent_t localizationDone, imageBufferCopied;

		// Intermediate data
		device_vec<float> d_com;
	};
	Batch* AllocBatch();
	void CopyBatchResults(Batch* b);

	enum { numThreads = 32 };
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

//	cudafft<float> *forward_fft, *backward_fft;

	Threads::Mutex currentBatchMutex, activeBatchMutex;
	std::vector<Batch*> freeBatches;
	std::list<Batch*> active;
	Batch* currentBatch;
	int maxActiveBatches;

	Threads::Mutex resultsMutex;
	std::vector<LocalizationResult> results;
	
	device_vec< sfft::complex<float> > fft_twiddles;
	device_vec< float2 > sharedBuf; // temp space for cpu mode or in case the hardware shared space is too small.
	device_vec< float2 > buffer; // general buffer space for computation
	int qiProfileLen, sharedMemSize; // QI profiles need to have power-of-two dimensions. qiProfileLen stores the closest power-of-two value that is bigger than cfg.qi_radialsteps
	cudaDeviceProp deviceProp;
	KernelParams kernelParams;

	int zlut_count, zlut_planes, zlut_res;
	cudaImageListf zlut;
	device_vec<float> zcompareWindow;

	void CallKernel_ComputeQI(cudaImageListf& images, QIParamWrapper params, uint sharedMem=0);
	void CallKernel_BgCorrectedCOM(cudaImageListf& images, float2* d_com, uint sharedMem=0);
	void CallKernel_MakeTestImage(cudaImageListf& images, float3* d_positions, uint sharedMem=0);

	int FetchResults();
	void QueueCurrentBatch();
};



