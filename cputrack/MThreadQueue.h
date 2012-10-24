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
	typedef typename TWorker::task_t Task;
	typedef typename TWorker::workspace_t Workspace;
	typedef typename TWorker::constructor_t Constructor;
	typedef typename TWorker::result_t Result;
	Constructor constructor;

	MThreadQueue(int workerThreads, Constructor wc)
	{
		constructor = wc;

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

	Workspace* GetFreeWorkspace() {
		Workspace* ws = 0;
		if (free_workspaces.empty()) {
			 ws = constructor.CreateWorkspace();
		} else {
			ws = free_workspaces.front();
			free_workspaces.pop_front();
		}
		return ws;
	}
	int GetQueueSize();

private:
	struct WorkerThread {
		pthread_t thread;
		bool active;
		TWorker* worker;
	};
	std::vector<WorkerThread> threads;
	pthread_mutex_t results_mutex;
	std::list<Result*> results;

	std::list<Task*> jobs;
	std::list<Workspace*> free_workspaces;
	pthread_mutex_t jobs_mutex;
	
	static void WorkerThreadMain(void* arg) 
	{
		WorkerThread* th = (WorkerThread*)arg;

		while (true) {
		}
	}

};

