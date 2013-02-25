// Simple memory tracker for debugging memory leaks
// Completed version of the one posted http://stackoverflow.com/questions/438515/how-to-track-memory-allocations-in-c-especially-new-delete
#ifdef USE_MEMDBG
#include <map>
#include <cstdint>

void dbgprintf(const char *fmt,...);

struct Allocation {
	const char *srcfile;
	int line;
	size_t size;
};

template<typename T>
struct track_alloc : std::allocator<T> {
    typedef typename std::allocator<T>::pointer pointer;
    typedef typename std::allocator<T>::size_type size_type;

    template<typename U>
    struct rebind {
        typedef track_alloc<U> other;
    };

    track_alloc() {}

    template<typename U>
    track_alloc(track_alloc<U> const& u)
        :std::allocator<T>(u) {}

    pointer allocate(size_type size, 
                     std::allocator<void>::const_pointer = 0) {
        void * p = std::malloc(size * sizeof(T));
        if(p == 0) {
            throw std::bad_alloc();
        }
        return static_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type) {
        std::free(p);
    }
};

typedef std::map< void*, Allocation, std::less<void*>, 
                  track_alloc< std::pair<void* const, std::size_t> > > AllocMap;

struct track_printer {
    AllocMap* track;
    track_printer(AllocMap * track):track(track) {}
    ~track_printer()
	{
		dbgprintf("Allocations: %d\n", track->size());
		size_t total = 0;
		for (AllocMap::iterator i = track->begin(); i != track->end(); ++i)
		{
			Allocation& a = i->second;
			dbgprintf("Allocation: %d bytes: @ line %d in '%s'\n" , a.size, a.line, a.srcfile);
			total += a.size;
		}
		dbgprintf("Total: %d bytes\n", total);
	}
};

AllocMap * get_map() {
    // don't use normal new to avoid infinite recursion.
    static AllocMap * track = new (std::malloc(sizeof *track)) AllocMap;
    static track_printer printer(track);
    return track;
}

void * operator new(size_t s, const char* file, int line) {
    // we are required to return non-null
    void * mem = std::malloc(s == 0 ? 1 : s);
    if(mem == 0) {
        throw std::bad_alloc();
    }
	Allocation alloc = { file, line ,s };
    (*get_map())[mem] = alloc;
    return mem;
}
void * operator new[](size_t s, const char* file, int line) {
    // we are required to return non-null
    void * mem = std::malloc(s == 0 ? 1 : s);
    if(mem == 0) {
        throw std::bad_alloc();
    }
	Allocation alloc = { file, line ,s };
    (*get_map())[mem] = alloc;
    return mem;
}

void operator delete(void * mem) {
    if(get_map()->erase(mem) == 0) {
        // this indicates a serious bug, or simply an STL allocation that is not calling new
//        dbgprintf("bug: memory at %p wasn't allocated by us\n", mem);
    }
    std::free(mem);
}
void operator delete[](void * mem) {
    if(get_map()->erase(mem) == 0) {
        // this indicates a serious bug
  //      dbgprintf("bug: memory at %p wasn't allocated by us\n", mem);
    }
    std::free(mem);
}


#endif
