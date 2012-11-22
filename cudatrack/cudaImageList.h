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
	CUBOTH bool boundaryHit(float2 center, float radius, int width, int height)
	{
		return center.x + radius >= width ||
			center.x - radius < 0 ||
			center.y + radius >= height ||
			center.y - radius < 0;
	}


	void free()
	{
		cudaFree(data);
	}

	int totalsize() { return pitch*h*count; }
};

