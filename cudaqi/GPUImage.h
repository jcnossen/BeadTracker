#pragma once
#ifndef _fundtypes_H
	#include <cstdint>
#endif

class GPUImage
{
public:
	GPUImage() { w=h=0; d_img=0; d_copyBuf=0; }
	GPUImage(int w, int h) { init(w,h); }
	GPUImage(int w) { init(w,w); }
	~GPUImage() {free();}

	void copyTo(GPUImage& dst)  const;
	float abssum(cudaStream_t s=0) const;
	void clear();
	void free();
	void init(int w,int h);

	int bytes() { return sizeof(float)*w*h; }
	int npixels() { return w*h; }

	int2 size() { 
		int2 s;
		s.x = w; s.y = h;
		return s;
	}
	void addfloat(float value, cudaStream_t s=0);
	float normalizeSum(float wantedSum, cudaStream_t s=0);
	float normalizeMax(float wantedMax, cudaStream_t s=0);
	float normalizeMinMax(float wantedMin, float wantedMax, cudaStream_t s=0);

	void clamp(float max, cudaStream_t s = 0);

	void resize(int w,int h);
	bool haveNaNs();

	float* d_ptr() { return d_img; }
	float const* d_ptr() const { return d_img; }

	int width() const { return w; }
	int height() const { return h; }

	void add(const GPUImage& src, cudaStream_t s=0);

	static GPUImage* buildFrom8bitStrided(uint8_t* data, int pixelsPerLine, int w,int h);
private:
	int w,h;
	float* d_img;

	uint8_t* d_copyBuf;
};
