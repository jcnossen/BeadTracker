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
	device_vec(bool emulate, size_t N) { 
		data = 0;
		size = 0;
		init(N);
//dbgprintf("%p. device_vec(emulate=%s, N=%d)\n",this, emulate?"true":"false", N);
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
	void copyToHost(std::vector<T>& dst ,bool async) {
		if (dst.size() != size)
			dst.resize(size);
		if (async)
			cudaMemcpyAsync(&dst[0], data, sizeof(T) * size, cudaMemcpyDeviceToHost);
		else
			cudaMemcpy(&dst[0], data, sizeof(T) * size, cudaMemcpyDeviceToHost);
	}
	void copyToDevice(std::vector<T>& src, bool async) {
		if (size != src.size())
			init(src.size());
		if (async)
			cudaMemcpyAsync(data, &src[0], sizeof(T) * size, cudaMemcpyHostToDevice);
		else
			cudaMemcpy(data, &src[0], sizeof(T) * size, cudaMemcpyHostToDevice);
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


