#pragma once
// Simple device side vector implementation. Thrust seems to end up in template errors, and I only need device_vector anyway.
#include <cuda_runtime.h>
#include <vector>
#include <cstdarg>

#include "simplefft.h"

#define LSQFIT_FUNC __device__ __host__
#include "LsqQuadraticFit.h"

#define CUBOTH __device__ __host__

#include "cudaImageList.h"

template<typename T>
class device_vec {
public:
	device_vec() {
	//	dbgprintf("%p device_vec()\n", this);
		data = 0;
		size = 0;
	}

	device_vec(size_t N) { 
		data = 0;
		size = 0;
		init(N);
//dbgprintf("%p. device_vec(emulate=%s, N=%d)\n",this, emulate?"true":"false", N);
	}
	device_vec(const device_vec<T>& src) {
		data = 0; size = 0;
		init(src.size);
	//	dbgprintf("%p. copy constructor: %p to %p. Host emulate=%d\n", this, src.data, data, host_emulate?1:0);
		cudaMemcpy(data, src.data, sizeof(T)*size, cudaMemcpyDeviceToDevice);
	}
	device_vec(const std::vector<T>& src) {
	//	dbgprintf("%p. operator=(vector)\n", this);
		data=0; size=0; 
		init(src.size());
		cudaMemcpy(data, &src[0], sizeof(T)*size, cudaMemcpyHostToDevice);
	}
	~device_vec(){
//dbgprintf("%p: ~device_vec. size=%d\n", this, size);
		cudaFree(data); 
		data=0;
	}
	void init(int s) {
		if(size != s) {
			clear();
			cudaMalloc(&data, sizeof(T)*s);
		}
		size = s;
	}
	void clear() {
		if (data) {
			cudaFree(data);
			data=0;
		}
	}
	operator std::vector<T>() const {
		std::vector<T> dst(size);
		cudaMemcpy(&dst[0], data, sizeof(T)*size, cudaMemcpyDeviceToHost);
		return dst;
	}
	device_vec<T>& operator=(const std::vector<T>& src) {
	//	dbgprintf("%p. operator=(vector)\n", this);
		init(src.size());
		cudaMemcpy(data, &src[0], sizeof(T)*size, cudaMemcpyHostToDevice);
		return *this;
	}
	device_vec<T>& operator=(const device_vec<T>& src) {
	//	dbgprintf("%p. operator=(device_vec)\n", this);
		clear();
		init(src.size);
		cudaMemcpy(data, src.data, sizeof(T)*size, cudaMemcpyDeviceToDevice);
		return *this;
	}
	void copyToHost(T* dst, bool async) {
		if (async)
			cudaMemcpyAsync(dst, data, sizeof(T) * size, cudaMemcpyDeviceToHost);
		else
			cudaMemcpy(dst, data, sizeof(T) * size, cudaMemcpyDeviceToHost);
	}
	void copyToHost(std::vector<T>& dst ,bool async) {
		if (dst.size() != size)
			dst.resize(size);
		copyToHost(&dst[0], async);
	}
	void copyToDevice(std::vector<T>& src, bool async) {
		copyToDevice(&src[0], src.size(), async);
	}
	void copyToDevice(T* first, int size, bool async) {
		if (this->size != size)
			init(size);
		if (async)
			cudaMemcpyAsync(data, first, sizeof(T) * size, cudaMemcpyHostToDevice);
		else
			cudaMemcpy(data, first, sizeof(T) * size, cudaMemcpyHostToDevice);
	}
	size_t size;
	T* data;
};



template<typename T>
CUBOTH T max_(T a, T b) { return a>b ? a : b; }
template<typename T>
CUBOTH T min_(T a, T b) { return a<b ? a : b; }

template<typename T, int numPts>
CUBOTH T ComputeMaxInterp(T* data, int len)
{
	int iMax=0;
	T vMax=data[0];
	for (int k=1;k<len;k++) {
		if (data[k]>vMax) {
			vMax = data[k];
			iMax = k;
		}
	}
	T xs[numPts]; 
	int startPos = max_(iMax-numPts/2, 0);
	int endPos = min_(iMax+(numPts-numPts/2), len);
	int numpoints = endPos - startPos;


	if (numpoints<3) 
		return iMax;
	else {
		for(int i=startPos;i<endPos;i++)
			xs[i-startPos] = i-iMax;

		LsqSqQuadFit<T> qfit(numpoints, xs, &data[startPos]);
		//printf("iMax: %d. qfit: data[%d]=%f\n", iMax, startPos, data[startPos]);
		//for (int k=0;k<numpoints;k++) {
	//		printf("data[%d]=%f\n", startPos+k, data[startPos]);
		//}
		T interpMax = qfit.maxPos();

		if (fabs(qfit.a)<1e-9f)
			return (T)iMax;
		else
			return (T)iMax + interpMax;
	}
}



#if 1 //defined(_DEBUG)
struct MeasureTime {
	uint64_t freq, time;
	const char* name;
	MeasureTime(const char *name) {
		QueryPerformanceCounter((LARGE_INTEGER*)&time);
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		this->name=name;
	}
	~MeasureTime() {
		uint64_t time1;
		QueryPerformanceCounter((LARGE_INTEGER*)&time1);
		double dt = (double)(time1-time) / (double)freq;
		dbgprintf("%s: Time taken: %f ms\n", name, dt*1000);
	}
};
#else
struct MeasureTime {
	MeasureTime(const char* name) {} 
};
#endif

//http://www.codeguru.com/cpp/cpp/cpp_mfc/stl/article.php/c4079/Allocators-STL.htm
template <class T, int flags=0> class cuda_host_allocator;

// specialize for void:
template <> class cuda_host_allocator<void> {
public:
    typedef void*       pointer;
    typedef const void* const_pointer;
    // reference to void members are impossible.
    typedef void value_type;
    template <class U> struct rebind { typedef cuda_host_allocator<U>
                                        other; };
};

template<typename T, int flags>
class cuda_host_allocator {
 
    public:
      typedef size_t    size_type;
      typedef ptrdiff_t difference_type;
      typedef T*        pointer;
      typedef const T*  const_pointer;
      typedef T&        reference;
      typedef const T&  const_reference;
      typedef T         value_type;
      template <class U> struct rebind { typedef cuda_host_allocator<U>
                                         other; };
 
      cuda_host_allocator() throw() { }
      cuda_host_allocator(const cuda_host_allocator&) throw() {  }
	  template <class U> cuda_host_allocator(const cuda_host_allocator<U>&) throw() {}
	  ~cuda_host_allocator() throw() {}
 
	  pointer address(reference x) const { return &x; }
	  const_pointer address(const_reference x) const { return &x; }
 
      pointer allocate(size_type n, cuda_host_allocator<void>::const_pointer hint = 0) {
		  pointer ptr;
		  cudaMallocHost(&ptr, sizeof(T)*n, flags);
		  return ptr;
	  }
      void deallocate(pointer p, size_type n) {
		  cudaFree(p);
	  }
	  size_type max_size() const throw() { return 500*1024*1024; }
 
      void construct(pointer p, const T& val) {
		new(static_cast<void*>(p)) T(val);
	  }
      void destroy(pointer p) {
		  p->~T();
	  }
  };

template <class T1, class T2> bool operator==(const cuda_host_allocator<T1>&, const cuda_host_allocator<T2>&) throw() { return true; }
template <class T1, class T2> bool operator!=(const cuda_host_allocator<T1>&, const cuda_host_allocator<T2>&) throw() { return false; }

template<typename T>
class pinned_vector : public std::vector<T, cuda_host_allocator<T> >
{
public:
	typedef std::vector<T, cuda_host_allocator<T> > base_t;
	pinned_vector(size_t N) : base_t(N) {}
	pinned_vector() {}
	pinned_vector(const pinned_vector<T>& o) : base_t(o) {}

};
