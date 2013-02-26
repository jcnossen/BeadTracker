#pragma once
#ifdef USE_MEMDBG
	void* operator new(size_t s, const char* file, int line);
	void* operator new[](size_t s, const char* file, int line);

	void operator delete(void* p);
	void operator delete[](void* p);

	#define new new(__FILE__, __LINE__)
	#pragma warning(disable: 4291) // no matching operator delete found; memory will not be freed if initialization throws an exception
	
#endif