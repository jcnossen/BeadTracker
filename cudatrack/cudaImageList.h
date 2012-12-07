#pragma once

#define CUBOTH __device__ __host__

template<typename T>
struct cudaImageList {
	// No constructor used to allow passing as CUDA kernel argument
	T* data;
	size_t pitch;
	int w,h;
	int count;

	static cudaImageList<T> alloc(int w,int h, int amount) {
		cudaImageList imgl;
		imgl.w = w; imgl.h = h;
		imgl.count = amount;
		cudaMallocPitch(&imgl.data, &imgl.pitch, sizeof(T)*w, h*amount);
		return imgl;
	}

	CUBOTH T* get(int i) {
		return (T*)(((char*)data) + pitch*h*i);
	}

	CUBOTH T& pixel(int x,int y, int imgIndex) {
		T* row = (T*) ( (char*)data + (h*imgIndex+y)*pitch );
		return row[x];
	}

	
	// Returns true if bounds are crossed
	CUBOTH bool boundaryHit(float2 center, float radius)
	{
		return center.x + radius >= w ||
			center.x - radius < 0 ||
			center.y + radius >= h ||
			center.y - radius < 0;
	}


	void free()
	{
		cudaFree(data);
	}

	CUBOTH int totalsize() { return pitch*h*count; }
	
	CUBOTH static inline T interp(T a, T b, float x) { return a + (b-a)*x; }

	CUBOTH T interpolate(float x,float y, int idx)
	{
		int rx=x, ry=y;

		T v00 = pixel(rx, ry, idx);
		T v10 = pixel(rx+1, ry, idx);
		T v01 = pixel(rx, ry+1, idx);
		T v11 = pixel(rx+1, ry+1, idx);

		T v0 = interp (v00, v10, x-rx);
		T v1 = interp (v01, v11, x-rx);

		return interp (v0, v1, y-ry);
	}

	void bind(texture<float, cudaTextureType2D, cudaReadModeElementType>& texref) {
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
		cudaBindTexture2D(NULL, &texref, data, &desc, w, h*count, pitch);
	}
	void unbind(texture<float, cudaTextureType2D, cudaReadModeElementType>& texref) {
		cudaUnbindTexture(&texref);
	}
};

