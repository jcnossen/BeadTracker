
#include <cuda_runtime.h>
#include <thrust/functional.h>

/* 
2D Array using CUDA pitched device memory 
*/
template<typename T>
class Array2D
{
public:
	size_t pitch, w, h;
	T* data;

	Array2D(size_t w,size_t h) {
		this->w = w;
		this->h = h;
		data = cudaMallocPitch(&data, &pitch, w, h);
	}

	~Array2D() {
		if (data)
			cudaFree(data);
	}

	Array2D(size_t w, size_t h, const T* host_data, size_t host_pitch=0) {
		this->w = w;
		this->h = h;
		cudaMallocPitch(&data, &pitch, w*sizeof(T), h);
		if (host_pitch==0) host_pitch = w*sizeof(T);
		cudaMemcpy2D(data, pitch, host_data, host_pitch, w, h, cudaMemcpyHostToDevice);
	}

	
};
