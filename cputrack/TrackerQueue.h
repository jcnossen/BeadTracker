#pragma once
#include "pthread.h"

struct TrackerLocalization {
	float x,y,z;
	int imageID;
	int param;
};

class TrackerQueue
{
public:
	TrackerQueue(int workerThreads, int width, int height, int xcorw);

	void QueueImage(unsigned short* data, int pitch, int imageID, int param);
	bool PollFinishedLocalization(TrackerLocalization *loc);

private:
	
};

