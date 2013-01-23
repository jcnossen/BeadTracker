// Thread OS related code is abstracted into a simple "Threads" struct
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

#include <stdexcept>
#include <Windows.h>
#undef AddJob
#undef Sleep
#undef max
#undef min


struct Threads
{
	struct Handle;
	struct Mutex {
		HANDLE h;
		Mutex() { h=CreateMutex(0,FALSE,0); }
		~Mutex() { CloseHandle(h); }
		void lock() { WaitForSingleObject(h, INFINITE); }
		void unlock() { ReleaseMutex(h); }
	};

	static Handle* Create(DWORD (WINAPI *method)(void* param), void* param) {
		DWORD threadID;
		HANDLE h = CreateThread(0, 0, method, param, 0, &threadID);
		if (!h) {
			throw std::runtime_error("Failed to create processing thread.");
		}
		return (Handle*)h;
	}

	static void WaitAndClose(Handle* h) {
		WaitForSingleObject((HANDLE)h, INFINITE);
		CloseHandle((HANDLE)h);
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
