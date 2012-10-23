#pragma once
#include "pthread.h"
#include <list>

/*
TWorker needs to have the following types declared:

constructor_t:  Functor to generate new workers
workspace_t:  Buffer type to maintain buffers shared between task processing
task_t:  Stores the task info
result_t:  Stores the result of the task
*/

template<typename TWorker>
class MThreadQueue
{
public:
	TWorker::constructor_t workerConstructor;

	MThreadQueue(int workerThreads, TWorker::constructor_t wc)
	{
		workerConstructor = wc;

		pthread_mutex_init(&results_mutex, 0);
		pthread_mutex_init(&jobs_mutex, 0);

		threads.resize(workerThreads);
		for (int k=0;k<workerThreads;k++) {
	//		pthread_create(& threads[k].thread, NULL, WorkerThreadMain);
		}

	}

	~MThreadQueue() {
		pthread_mutex_destroy(&results_mutex);
		pthread_mutex_destroy(&jobs_mutex);
	}

	TTaskWorkspace* GetTaskWorkspace() {
		if (free_workspace.empty()) {
			TTaskWorkspace * ws = 
		}
	}
	void QueueTask(TTask* task);
	bool PollFinishedLocalization(TrackerLocalization *loc);
	int GetQueueSize();

private:
	Job* AllocateJob();

	struct WorkerThread {
		pthread_t thread;
		bool active;

	};
	std::vector<WorkerThread> threads;
	pthread_mutex_t results_mutex;
	std::list<TResult> results;

	std::list<Job*> jobs;
	std::list<TTaskWorkspace*> free_workspace;
	pthread_mutex_t jobs_mutex;

	TWorkerParam *workerParam;

	static void WorkerThreadMain(void* arg) 
	{
		WorkerThread* th = (WorkerThread*)arg;
	}




};

