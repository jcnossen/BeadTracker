#pragma once
#include "threads.h"
#include "QueuedTracker.h"
#include <cuda_runtime_api.h>
#include "cudafft/cudafft.h"
#include <list>
#include "cudaImageList.h"

template<typename T>
struct cudaImageList;
typedef cudaImageList<float> cudaImageListf;

class QueuedCUDATracker : public QueuedTracker {
public:
	QueuedCUDATracker(QTrkSettings* cfg);
	~QueuedCUDATracker();

	void Start();

	void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, vector3f* initialPos, uint zlutIndex=0);
	int PollFinished(LocalizationResult* results, int maxResults);

	void SetZLUT(float* data, int planes, int res, int numLUTs);
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
	void Compute1DXCorProfiles(cudaImageListf& images, float* d_profiles);

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
		
		texture<float, cudaTextureType2D, cudaReadModeElementType> texref;
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

	cudafft<float> *forward_fft, *backward_fft;
	float2* xcor_workspace;

	Threads::Mutex batch_mutex;
	std::vector<Batch*> freeBatches;
	std::vector<Job*> jobs;

	Threads::Mutex batchMutex;
	std::vector<Batch*> active;
};






