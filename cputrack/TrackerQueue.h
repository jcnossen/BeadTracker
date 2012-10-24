#include "Tracker.h"
#include "MThreadQueue.h"

struct LocalizationResult {
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
	struct LocalizationTask {
		TrackerImageBuffer *imageBuffer;
	};

/*
constructor_t:  Functor to generate new workers
workspace_t:  Buffer type to maintain buffers shared between task processing
task_t:  Stores the task info
result_t:  Stores the resultof the task
*/
	struct LocalizationWorker;

	struct NewWorkerFunctor {
		LocalizationWorker* operator()() { return new LocalizationWorker(); }
	};

	struct LocalizationWorker {
		// MThreadQueue type interface
		typedef NewWorkerFunctor constructor_t;
		typedef TrackerImageBuffer workspace_t;
		typedef LocalizationTask task_t;
		typedef LocalizationResult result_t;
	};

	//MThreadQueue<LocalizationWorker> threadQueue;
};
