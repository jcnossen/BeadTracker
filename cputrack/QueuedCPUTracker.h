#pragma once

#include "threads.h"

#include "QueuedTracker.h"
#include "cpu_tracker.h"

#include <list>

class QueuedCPUTracker : public QueuedTracker {
public:
	QueuedCPUTracker(const QTrkComputedConfig& cc);
	~QueuedCPUTracker();
	void Start();
	void Break(bool pause);
	void GenerateTestImage(float* dst, float xp, float yp, float z, float photoncount);
	int NumThreads() { return cfg.numThreads; }

	// QueuedTracker interface
	void SetZLUT(float* data, int num_zluts, int planes, float* zcmp=0) override;
	float* GetZLUT(int *num_zluts, int* planes) override;
	void ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo) override;
	// Schedule an entire frame at once, allowing for further optimizations
	void ScheduleFrame(uchar *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, 
		const LocalizationJob *jobInfo) override;
	
	int GetQueueLength(int *maxQueueLength=0) override; // In queue + in progress
	int PollFinished(LocalizationResult* results, int maxResults) override;
	void ClearResults() override;
	void Flush() override { };

	bool IsIdle() override;
	int GetResultCount() override;

	std::string GetProfileReport() { return "CPU tracker currently has no profile reporting"; }

private:
	struct Thread {
		Thread() { tracker=0; manager=0; thread=0;}
		CPUTracker *tracker;
		Threads::Handle* thread;
		QueuedCPUTracker* manager;
	};

	struct Job {
		Job() { data=0; dataType=QTrkU8; }
		~Job() { delete[] data; }

		uchar* data;
		QTRK_PixelDataType dataType;
		LocalizationJob job;
	};

	// Special no-threads mode for debugging
	CPUTracker* noThreadTracker;

	Threads::Mutex jobs_mutex, jobs_buffer_mutex, results_mutex;
	std::deque<Job*> jobs;
	int jobCount;
	std::vector<Job*> jobs_buffer; // stores memory
	std::deque<LocalizationResult> results;
	int resultCount;
	int maxQueueSize;
	int jobsInProgress;

	std::vector<Thread> threads;
	float* zluts;
	int zlut_count, zlut_planes;
	std::vector<float> zcmp;
	float* GetZLUTByIndex(int index) { return &zluts[ index * (zlut_planes*cfg.zlut_radialsteps) ]; }
	void UpdateZLUTs();

	// signal threads to stop their work
	bool quitWork, processJobs;

	void JobFinished(Job* j);
	Job* GetNextJob();
	Job* AllocateJob();
	void AddJob(Job* j);
	void ProcessJob(CPUTracker* trk, Job* j);

	static void WorkerThreadMain(void* arg);
};

