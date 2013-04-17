#pragma once

template<typename T>
struct cudaImageList {
	// No constructor used to allow passing as CUDA kernel argument
	T* data;
	size_t pitch;
	int w,h;
	int count;

	static cudaImageList<T> empty() {
		cudaImageList imgl;
		imgl.data = 0;
		imgl.pitch = 0;
		imgl.w = imgl.h = imgl.count = 0;
		return imgl;
	}

	static cudaImageList<T> alloc(int w,int h, int amount) {
		cudaImageList imgl;
		imgl.w = w; imgl.h = h;
		imgl.count = amount;
		if (cudaMallocPitch(&imgl.data, &imgl.pitch, sizeof(T)*w, h*amount) != cudaSuccess) {
			throw std::bad_alloc(SPrintf("cudaImageListf<%s> alloc %dx%dx%d failed", typeid(T).name(), w, h, amount).c_str());
		}
		return imgl;
	}

	CUBOTH T* get(int i) {
		return (T*)(((char*)data) + pitch*h*i);
	}

	CUBOTH T pixel_oobcheck(int x,int y, int imgIndex, T border=0.0f) {
		if (x < 0 || x >= w || y < 0 || y >= h)
			return border;

		computeImagePos(x,y,imgIndex);
		T* row = (T*) ( (char*)data + y*pitch );
		return row[x];
	}

	CUBOTH T pixel(int x,int y, int imgIndex) {
		computeImagePos(x,y,imgIndex);
		T* row = (T*) ( (char*)data + y*pitch );
		return row[x];
	}

	CUBOTH T* pixelAddress(int x,int y, int imgIndex) {
		if (x < 0 || x >= w || y < 0 || y >= h)
			return 0;

		computeImagePos(x,y,imgIndex);
		T* row = (T*) ( (char*)data + y*pitch );
		return row + x;
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
		if(data) cudaFree(data);
	}

	void copyToHost(T* dst, bool async) {
		if (async)
			cudaMemcpy2DAsync(dst, sizeof(T)*w, data, pitch, w*sizeof(T), count*h, cudaMemcpyDeviceToHost);
		else
			cudaMemcpy2D(dst, sizeof(T)*w, data, pitch, w*sizeof(T), count*h, cudaMemcpyDeviceToHost);
	}
	
	void copyToDevice(T* src, bool async) {
		if (async)
			cudaMemcpy2DAsync(data, pitch, src, w*sizeof(T), w*sizeof(T), count*h, cudaMemcpyHostToDevice);
		else
			cudaMemcpy2D(data, pitch, src, w*sizeof(T), w*sizeof(T), count*h, cudaMemcpyHostToDevice);
	}

	void clear() {
		cudaMemset2D(data, pitch, 0, w*sizeof(T), count*h);
	}

	CUBOTH int totalNumPixels() { return pitch*h*count; }
	CUBOTH int totalNumBytes() { return pitch*h*count*sizeof(T); }
	
	CUBOTH static inline T interp(T a, T b, float x) { return a + (b-a)*x; }

	CUBOTH T interpolate(float x,float y, int idx, float border=0.0f)
	{
		int rx=x, ry=y;

		if (rx < 0 || ry < 0 || rx >= w-1 || ry >= h-1)
			return border;

		T v00 = pixel(rx, ry, idx);
		T v10 = pixel(rx+1, ry, idx);
		T v01 = pixel(rx, ry+1, idx);
		T v11 = pixel(rx+1, ry+1, idx);

		T v0 = interp (v00, v10, x-rx);
		T v1 = interp (v01, v11, x-rx);

		return interp (v0, v1, y-ry);
	}

	void bind(texture<T, cudaTextureType2D, cudaReadModeElementType>& texref) {
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
		cudaBindTexture2D(NULL, &texref, data, &desc, w, h*count, pitch);
	}
	void unbind(texture<T, cudaTextureType2D, cudaReadModeElementType>& texref) {
		cudaUnbindTexture(&texref);
	}

	CUBOTH void computeImagePos(int& x, int& y, int idx)
	{
		y += idx * h;
	}

	// Using the texture cache can result in significant speedups
	__device__ T interpolateFromTexture(texture<T, cudaTextureType2D, cudaReadModeElementType> texref, float x,float y, int idx, float border=0.0f)
	{
		int rx=x, ry=y;

		if (rx < 0 || ry < 0 || rx >= w-1 || ry >= h-1)
			return border;

		computeImagePos(rx, ry, idx);

		float fx=x-floor(x), fy = y-floor(y);
		float u = rx + 0.5f;
		float v = ry + 0.5f;

		T v00 = tex2D(texref, u, v);
		T v10 = tex2D(texref, u+1, v);
		T v01 = tex2D(texref, u, v+1);
		T v11 = tex2D(texref, u+1, v+1);

		T v0 = interp (v00, v10, fx);
		T v1 = interp (v01, v11, fx);

		return interp (v0, v1, fy);
	}
};

