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
	void SetZLUT(float* data, int num_zluts, int planes, int res);
	float* GetZLUT(int* num_zluts,int *planes, int* res);
	
	void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, vector3f* initialPos, uint zlutIndex=0, uint zlutPlane=0);
	int PollFinished(LocalizationResult* results, int maxResults);
	void ClearResults();
	void GenerateTestImage(float* dst, float xp,float yp, float z, float photoncount);

	int GetJobCount();
	int GetResultCount();
	int NumThreads() { return 1; }

private:
	float* GetZLUTByIndex(int index) { return &zluts[ index * (zlut_planes*zlut_res) ]; }

	std::list<LocalizationResult> results;
	int resultCount;
	CPUTracker *tracker;

	float* zluts;
	int zlut_count, zlut_planes, zlut_res;
};

