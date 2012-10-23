#pragma once

#include "../cudatrack/utils.h"
#include "TrackerQueue.h"


TrackerQueue::TrackerQueue(int workerThreads, int width, int height, int xcorw, ZLookupTable* zlut)
{
	results_mutex = 0;
	jobs_mutex = 0;

	threads.resize(workerThreads);
	for (int k=0;k<workerThreads;k++) {
//		pthread_create(& threads[k].thread, NULL, WorkerThreadMain);
	}

}


void TrackerQueue::QueueImage(unsigned short* data, int pitch, int imageID, int param)
{
	Job* job = AllocateJob();

	job->buffer->Assign(data, pitch);
}


TrackerQueue::Job* TrackerQueue::AllocateJob()
{
	TrackerQueue::Job* job;
	if(job_buffer.empty()) {
		job = new TrackerQueue::Job();
		job->buffer = CreateTrackerImageBuffer(trk_w, trk_h);
	}
	else {
		job = job_buffer.front();
		job_buffer.pop_front();
	}
	return job;
}



