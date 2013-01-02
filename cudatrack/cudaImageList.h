#pragma once

template<typename T>
struct cudaImageList {
	// No constructor used to allow passing as CUDA kernel argument
	T* data;
	size_t pitch;
	int w,h;
	int count;
	bool hostMem;
	T borderValue; // value that pixel() returns outside of image

	static cudaImageList<T> alloc(int w,int h, int amount, bool hostMem) {
		cudaImageList imgl;
		imgl.w = w; imgl.h = h;
		imgl.count = amount;
		if (hostMem) {
			imgl.data = new T[w*h*amount];
			imgl.pitch = w*sizeof(T);
		} else
			cudaMallocPitch(&imgl.data, &imgl.pitch, sizeof(T)*w, h*amount);
		imgl.hostMem = hostMem;
		imgl.borderValue = 0.0;
		return imgl;
	}

	CUBOTH T* get(int i) {
		return (T*)(((char*)data) + pitch*h*i);
	}

	CUBOTH T& pixel(int x,int y, int imgIndex) {
		if (x < 0 || x >= w || y < 0 || y >= h)
			return borderValue;

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
		if(hostMem)
			delete[] data;
		else
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

