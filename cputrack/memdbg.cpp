#include <map>
#include <cstdint>
#undef new

void dbgprintf(const char *fmt,...);


struct Allocation {
	const char *srcfile;
	int line;
	size_t size;
};

typedef std::map<uint64_t, Allocation> AllocMap;
static AllocMap alloc_map;

void* operator new(size_t s, const char* file, int line)
{
	void *mem = malloc(s);

	Allocation alloc = { file, line, s };
	alloc_map[(uint64_t)mem] = alloc;
	return mem;
}
void* operator new[](size_t s, const char* file, int line)
{
	void *mem = malloc(s);

	Allocation alloc = { file, line, s };
	alloc_map[(uint64_t)mem] = alloc;
	return mem;
}

void MemDbgUnregister(void *p)
{
	alloc_map.erase((uint64_t)p);
}


void MemDbgListAllocations()
{
	dbgprintf("Allocations: %d\n", alloc_map.size());
	size_t total = 0;
	for (AllocMap::iterator i = alloc_map.begin(); i != alloc_map.end(); ++i)
	{
		Allocation& a = i->second;
		dbgprintf("Allocation: %d bytes: @ line %d in '%s'\n" , a.size, a.line, a.srcfile);
		total += a.size;
	}
	dbgprintf("Total: %d bytes\n", total);

	alloc_map.clear();
}

