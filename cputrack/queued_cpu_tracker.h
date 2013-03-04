#pragma once

#include "threads.h"

#include "QueuedTracker.h"
#include "cpu_tracker.h"

#include <list>

class QueuedCPUTracker : public QueuedTracker {
public:
	QueuedCPUTracker(QTrkSettings* settings);
	~QueuedCPUTracker();

	void Start();
	void Break(bool pause);
	void SetZLUT(float* data, int num_zluts, int planes, int res, float* zcmp);
	float* GetZLUT(int *num_zluts, int* planes, int* res);
	bool ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane);
	// Schedule an entire frame at once, allowing for further optimizations
	void ScheduleFrame(uchar *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, 
		LocalizeType locType, uint frame, uint zlutPlane, bool async);
	void WaitForScheduleFrame(uchar* imgptr) {} // CPU tracker does not support asynchronous calls to ScheduleFrame
	
	int PollFinished(LocalizationResult* results, int maxResults);
	void ClearResults();
	void GenerateTestImage(float* dst, float xp, float yp, float z, float photoncount);
	void Flush() { }

	bool IsQueueFilled() { return GetJobCount() >= cfg.maxQueueSize; }
	bool IsIdle() { return GetJobCount() == 0; }

	int GetResultCount();
	int GetJobCount();
	int NumThreads() { return cfg.numThreads; }

private:
	struct Thread {
		Thread() { tracker=0; manager=0; thread=0;}
		CPUTracker *tracker;
		Threads::Handle* thread;
		QueuedCPUTracker* manager;
	};

	struct Job {
		Job() { data=0; dataType=QTrkU8; locType=LocalizeXCor1D; id=0; zlut=0; initialPos.x=initialPos.y=initialPos.z=0.0f; zlutPlane=0; }
		~Job() { delete[] data; }

		uchar* data;
		QTRK_PixelDataType dataType;
		LocalizeType locType;
		uint id;
		uint zlut, zlutPlane;
		vector3f initialPos;
	};

	// Special no-threads mode for debugging
	CPUTracker* noThreadTracker;

	Threads::Mutex jobs_mutex, jobs_buffer_mutex, results_mutex;
	std::list<Job*> jobs;
	int jobCount;
	std::vector<Job*> jobs_buffer; // stores memory
	std::vector<LocalizationResult> results;
	int resultCount;

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

	static DWORD WINAPI WorkerThreadMain(void* arg);
};

