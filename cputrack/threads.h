
#include <stdexcept>
#include <Windows.h>
#undef AddJob
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
};

typedef Threads::Handle ThreadHandle;


