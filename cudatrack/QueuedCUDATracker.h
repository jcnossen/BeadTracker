#pragma once
#include "QueuedTracker.h"
#include <cuda_runtime_api.h>
#include "cudafft/cudafft.h"

template<typename T>
struct cudaImageList;
typedef cudaImageList<float> cudaImageListf;

class QueuedCUDATracker : public QueuedTracker {
public:
	QueuedCUDATracker(QTrkSettings* cfg);
	~QueuedCUDATracker();

	void Start();

	void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, uint zlutIndex=0);
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

protected:
	QTrkSettings cfg;

	struct Batch {

//		texture<

	};

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
};






