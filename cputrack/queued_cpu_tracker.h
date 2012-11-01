#pragma once
#include "pthread.h"
#include "QueuedTracker.h"
#include "cpu_tracker.h"


class QueuedCPUTracker : public QueuedTracker {
public:
	QueuedCPUTracker(int numThreads);
	~QueuedCPUTracker();

	void SetZLUT(float* data, int planes, int res, int num_zluts, float prof_radius, int angularSteps);
	float ComputeZ(vector2f center, int angularSteps, int zlutIndex); // radialSteps is given by zlut_res

	void ScheduleLocalization(TrackerImageBuffer* buffer, Localize2DType locType, bool computeZ, uint id, uint zlutIndex=0);
	int PollFinished(LocalizationResult* results, int maxResults);

	struct Thread {
		CPUTracker *tracker;
		pthread_t thread;
	};

	struct Job {
	};

	pthread_mutex_t jobs_mutex;
	std::vector<Job> jobs;

	std::vector<Thread> threads;
	float* zluts;
};
