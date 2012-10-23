#pragma once
#include "pthread.h"
#include <list>

#include "Tracker.h"

struct TrackerLocalization {
	float x,y,z;
	int imageID;
	int param;
};



struct ZLookupTable {
	ZLookupTable(float* d, int planes, int res, float radius) { 
		data=d;
		this->planes = planes;
		radialsteps = res;
		profile_radius = radius;
	}
	~ZLookupTable() { if(data) delete[] data; }
	float* data;
	int planes, radialsteps;
	float profile_radius;
};
 

class TrackerQueue
{
public:
	TrackerQueue(int workerThreads, int width, int height, int xcorw, ZLookupTable* zlut);
	~TrackerQueue();

	void QueueImage(unsigned short* data, int pitch, int imageID, int param);
	bool PollFinishedLocalization(TrackerLocalization *loc);
	int GetQueueSize();

private:
	struct Job {
		TrackerImageBuffer* buffer;
		int imageID, param;
	};

	Job* AllocateJob();

	struct WorkerThread {
		pthread_t thread;
		Tracker* tracker;
		TrackerQueue* queue;
	};
	std::vector<WorkerThread> threads;
	pthread_mutex_t results_mutex;
	std::list<TrackerLocalization> results;

	std::list<Job*> jobs;
	std::list<Job*> job_buffer;
	pthread_mutex_t jobs_mutex;

	int trk_w, trk_h, trk_xcorw;
	ZLookupTable* trk_zlut;

	static void WorkerThreadMain(void* arg);
};

