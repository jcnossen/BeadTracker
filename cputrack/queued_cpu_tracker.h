#pragma once
#include "pthread.h"
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
	int NumThreads() { return cfg.numThreads; }

private:
	struct Thread {
		Thread() { tracker=0; manager=0; }
		CPUTracker *tracker;
		pthread_t thread;
		QueuedCPUTracker* manager;
	};

	struct Job {
		Job() { data=0; dataType=QTrkU8; locType=LocalizeXCor1D; id=0; zlut=0;  }
		~Job() { delete[] data; }

		uchar* data;
		QTRK_PixelDataType dataType;
		LocalizeType locType;
		uint id;
		uint zlut;
	};

	pthread_attr_t joinable_attr;
	pthread_mutex_t jobs_mutex, jobs_buffer_mutex, results_mutex;
	std::list<Job*> jobs;
	int jobCount;
	std::vector<Job*> jobs_buffer; // stores memory
	std::list<LocalizationResult> results;
	int resultCount;

	std::vector<Thread> threads;
	float* zluts;
	int zlut_count, zlut_planes, zlut_res;

	// signal threads to stop their work
	bool quitWork;

	void JobFinished(Job* j);
	Job* GetNextJob();
	Job* AllocateJob();
	void AddJob(Job* j);
	void ProcessJob(Thread* th, Job* j);

	static void* WorkerThreadMain(void* arg);
};
