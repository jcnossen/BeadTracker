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
			if (s!=0) cudaMalloc(&data, sizeof(T)*s);
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
	void copyToHost(T* dst, bool async, cudaStream_t s=0) {
		if (async)
			cudaMemcpyAsync(dst, data, sizeof(T) * size, cudaMemcpyDeviceToHost, s);
		else
			cudaMemcpy(dst, data, sizeof(T) * size, cudaMemcpyDeviceToHost);
	}
	void copyToHost(std::vector<T>& dst ,bool async, cudaStream_t s=0) {
		if (dst.size() != size)
			dst.resize(size);
		copyToHost(&dst[0], async, s);
	}
	void copyToDevice(std::vector<T>& src, bool async, cudaStream_t s=0) {
		copyToDevice(&src[0], src.size(), async, s);
	}
	void copyToDevice(T* first, int size, bool async, cudaStream_t s=0) {
		if (this->size != size)
			init(size);
		if (async)
			cudaMemcpyAsync(data, first, sizeof(T) * size, cudaMemcpyHostToDevice, s);
		else
			cudaMemcpy(data, first, sizeof(T) * size, cudaMemcpyHostToDevice);
	}
	size_t memsize() { return size*sizeof(T); }
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


template<typename T, int flags=0>
class pinned_array
{
public:
	pinned_array() {
		d=0; n=0;
	}
	~pinned_array() {
		free();
	}
	pinned_array(size_t n) {
		d=0; init(n);
	}
	template<typename TOther, int f>
	pinned_array(const pinned_array<TOther,f>& src) {
		d=0;init(src.n);
		for(int k=0;k<src.n;k++)
			d[k]=src[k];
	}
	template<typename TOther, int F>
	pinned_array& operator=(const pinned_array<TOther, F>& src) {
		if (src.n != n) init(src.n);
		for(int k=0;k<src.n;k++)
			d[k]=src[k];
		return *this;
	}
	template<typename Iterator>
	pinned_array(Iterator first, Iterator end) {
		d=0; init(end-first);
		for (int k = 0; first != end; ++first) {
			d[k++] = *first;
		}
	}
	template<typename T>
	pinned_array(const device_vec<T>& src) {
		d=0; init(src.size()); src.copyToHost(d,false);
	}

	int size() const { return n; }
	T* begin() { return d; }
	T* end() { return d+n; }
	const T* begin() const { return d; }
	const T* end() const { return d+n; }
	T* data() { return d; }
	void free() {
		cudaFreeHost(d);
		d=0;n=0;
	}
	void init(int n) {
		if (d) free();
		this->n = n;
		cudaMallocHost(&d, sizeof(T)*n, flags);
	}
	T& operator[](int i) {  return d[i]; }
	const T&operator[](int i) const { return d[i];}
	size_t memsize() { return n*sizeof(T); }

protected:
	T* d;
	size_t n;
};
