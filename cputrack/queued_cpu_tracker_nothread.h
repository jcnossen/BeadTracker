#pragma once
#include <Windows.h>
#undef AddJob
#undef max
#undef min

#include "QueuedTracker.h"
#include "cpu_tracker.h"

#include <list>

class QueuedCPUTracker : public QueuedTracker {
public:
	QueuedCPUTracker(QTrkSettings* settings);
	~QueuedCPUTracker();

	void Start();
	void SetZLUT(float* data, int planes, int res, int num_zluts);
	void ComputeRadialProfile(float *image, int width, int height, float* dst, int profileLen, vector2f center);

	void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, uint zlutIndex=0);
	int PollFinished(LocalizationResult* results, int maxResults);

	void GenerateTestImage(float* dst, float xp,float yp, float z, float photoncount);

	int GetJobCount();
	int GetResultCount();
	int NumThreads() { return 1; }

private:
	std::list<LocalizationResult> results;
	int resultCount;
	CPUTracker *tracker;

	float* zluts;
	int zlut_count, zlut_planes, zlut_res;
};

