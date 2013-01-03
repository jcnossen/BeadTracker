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
	float2* d_initial;
	float2* d_output;
	sfft::complex<float>* d_twiddles;
	uchar* d_boundaryHits;
	float minRadius, maxRadius;
	int radialSteps, iterations, angularSteps;
	bool useShared;
	float2* sharedBuf;
};


class QueuedCUDATracker : public QueuedTracker {
public:
	QueuedCUDATracker(QTrkSettings* cfg);
	~QueuedCUDATracker();

	void Start();
	void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane);
	void ClearResults();

	void SetZLUT(float* data,  int numLUTs, int planes, int res); // data can be zero to allocate ZLUT data
	float* GetZLUT(int* planes=0, int *res=0, int *count=0); // delete[] memory afterwards
	int PollFinished(LocalizationResult* results, int maxResults);

	void ComputeRadialProfile(float *image, int width, int height, float* dst, int profileLength, vector2f center);
	void GenerateTestImage(float* dst, float xp,float yp, float z, float photoncount);

	// Debug stuff
	float* GetDebugImage() { return 0; }

	int GetJobCount();
	int GetResultCount();

	// Direct kernel wrappers
	void GenerateImages(cudaImageListf& imgList, float3 *d_positions);
	void ComputeBgCorrectedCOM(cudaImageListf& imgList, float2* d_com);
	void Compute1DXCor(cudaImageListf& images, float2* d_initial, float2* d_result);
	void ComputeQI(cudaImageListf& images, float2* d_initial, float2* d_result);
	void Compute1DXCorProfiles(cudaImageListf& images, float* d_profiles);

	template<typename T>
	device_vec<T> DeviceMem(int size=0) { 
		return device_vec<T>(useCPU, size);
	}
	template<typename T>
	device_vec<T> DeviceMem(const std::vector<T>& src) {
		device_vec<T> d(useCPU);
		d = src;
		return d;
	}
	bool UseHostEmulate() { return useCPU; } // true if we run all the kernels on the CPU side, for debugging

protected:
	QTrkSettings cfg;

	struct Job {
		Job() { locType=LocalizeXCor1D; id=0; zlut=0; initialPos.x=initialPos.y=initialPos.z=0.0f; }

		LocalizeType locType;
		uint id;
		uint zlut;
		vector3f initialPos;
	};

	struct Batch{
		Batch() { d_profiles = 0; hostImageBuf = 0; imageBuf.data=0; }
		~Batch();
		
		float* d_profiles;
		std::vector<Job> jobs;
		cudaImageListf imageBuf;
		float* hostImageBuf;
	};
	Batch* AllocBatch();

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

	std::vector<Batch*> freeBatches;
	std::vector<Job*> jobs;

	Threads::Mutex batchMutex;
	std::vector<Batch*> active;

	device_vec< sfft::complex<float> > fft_twiddles;
	bool useCPU;
	device_vec< float2 > sharedBuf; // temp space for cpu mode or in case the hardware shared space is too small.
	int qiProfileLen; // QI profiles need to have power-of-two dimensions. qiProfileLen stores the closest power-of-two value that is bigger than cfg.qi_radialsteps
	cudaDeviceProp deviceProp;

	void CallKernel_BgCorrectedCOM(cudaImageListf& images, float2* d_com, uint sharedMem=0);
	void CallKernel_MakeTestImage(cudaImageListf& images, float3* d_positions, uint sharedMem=0);
	void CallKernel_ComputeQI(cudaImageListf& images, QIParams d_params, uint sharedMem=0);
};



