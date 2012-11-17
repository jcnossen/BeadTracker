#pragma once
#include "QueuedTracker.h"

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

protected:
	QTrkSettings cfg;
};



