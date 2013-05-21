// Thread OS related code is abstracted into a simple "Threads" struct
#pragma once
#ifdef USE_PTHREADS

#include "pthread.h"

struct Threads
{
	struct Handle;

	pthread_attr_t joinable_attr;

	struct Mutex {
		pthread_mutex_t h;
		Mutex() { pthread_mutex_init(&h, 0);  }
		~Mutex() { pthread_mutex_destroy(&h);  }
		void lock() { 
			pthread_mutex_lock(&h); }
		void unlock() { pthread_mutex_unlock(&h); }
	};

	static Handle* Create(DWORD (WINAPI *method)(void* param), void* param) {
		pthread_t h;
		pthread_attr_t joinable_attr;
		pthread_attr_init(&joinable_attr);
		pthread_attr_setdetachstate(&joinable_attr, PTHREAD_CREATE_JOINABLE);
		pthread_create(&h, &joinable_attr, method, param);
		if (!h) {
			throw std::runtime_error("Failed to create processing thread.");
		}

		pthread_attr_destroy(&joinable_attr);
		return (Handle*)h;
	}

	static void WaitAndClose(Handle* h) {
		pthread_join((pthread_t)h, 0);
	}
};


#else

#include <Windows.h>
#undef AddJob
#undef Sleep
#undef max
#undef min


struct Threads
{
	typedef void (*ThreadEntryPoint)(void* param);
	struct Handle {
		DWORD threadID;
		ThreadEntryPoint callback;
		HANDLE winhdl;
		void* param;
	};

	struct Mutex {
		HANDLE h;
		Mutex() { h=CreateMutex(0,FALSE,0); }
		~Mutex() { CloseHandle(h); }
		void lock() { WaitForSingleObject(h, INFINITE); }
		void unlock() { ReleaseMutex(h); }
	};

	static DWORD WINAPI ThreadCaller (void *param) {
		Handle* hdl = (Handle*)param;
		hdl->callback (hdl->param);
		return 0;
	}

	static Handle* Create(ThreadEntryPoint method, void* param) {
		Handle* hdl = new Handle;
		hdl->param = param;
		hdl->callback = method;
		hdl->winhdl = CreateThread(0, 0, ThreadCaller, hdl, 0, &hdl->threadID);
		
		if (!hdl->winhdl) {
			throw std::runtime_error("Failed to create processing thread.");
		}
		return hdl;
	}

	static bool RunningVistaOrBetter ()
	{
		OSVERSIONINFO v;
		GetVersionEx(&v);
		return v.dwMajorVersion >= 6;
	}

	static void SetBackgroundPriority(Handle* thread, bool bg)
	{
		HANDLE h = (HANDLE)thread;
		// >= Windows Vista
		if (RunningVistaOrBetter())
			SetThreadPriority(h, bg ? THREAD_MODE_BACKGROUND_BEGIN : THREAD_MODE_BACKGROUND_END);
		else
			SetThreadPriority(h, bg ? THREAD_PRIORITY_BELOW_NORMAL : THREAD_PRIORITY_NORMAL);
	}

	static void WaitAndClose(Handle* h) {
		WaitForSingleObject(h->winhdl, INFINITE);
		CloseHandle(h->winhdl);
		delete h;
	}

	static void Sleep(int ms) {
		::Sleep(ms);
	}

	static int GetCPUCount() {
		// preferably 
		#ifdef WIN32
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		return sysInfo.dwNumberOfProcessors;
		#else
		return 4;
		#endif
	}
};

typedef Threads::Handle ThreadHandle;


#endif
