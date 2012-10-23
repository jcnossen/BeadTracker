#include "Tracker.h"
#include "MThreadQueue.h"

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
	struct LocalizationTask {
		TrackerImageBuffer *imageBuffer;
	};

	MThreadQueue<TrackerQueue, LocalizationTask, TrackerLocalization> threadQueue;
};
