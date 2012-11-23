#pragma once

#define CUBOTH __device__ __host__

struct cudaImageList {
	// No constructor used to allow passing as CUDA kernel argument
	float* data;
	size_t pitch;
	int w,h;
	int count;

	static cudaImageList alloc(int w,int h, int amount) {
		cudaImageList imgl;
		imgl.w = w; imgl.h = h;
		imgl.count = amount;
		cudaMallocPitch(&imgl.data, &imgl.pitch, sizeof(float)*w, h*amount);
		return imgl;
	}

	CUBOTH float* get(int i) {
		return (float*)(((char*)data) + pitch*h*i);
	}

	CUBOTH float& pixel(int x,int y, int imgIndex) {
		float* row = (float*) ( (char*)data + (h*imgIndex+y)*pitch );
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
	
	CUBOTH static inline float interp(float a, float b, float x) { return a + (b-a)*x; }

	CUBOTH float interpolate(float x,float y, int idx)
	{
		int rx=x, ry=y;

		float v00 = pixel(rx, ry, idx);
		float v10 = pixel(rx+1, ry, idx);
		float v01 = pixel(rx, ry+1, idx);
		float v11 = pixel(rx+1, ry+1, idx);

		float v0 = interp (v00, v10, x-rx);
		float v1 = interp (v01, v11, x-rx);

		return interp (v0, v1, y-ry);
	}
};

