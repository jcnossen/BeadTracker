#pragma once
#include "cudaImageList.h"

class ImageSampler_MemCopy {
public:
	static void BindTexture(cudaImageListf& images) { }
	static void UnbindTexture(cudaImageListf& images) { }

	// All interpolated texture/images fetches go through here
	static __device__ float Interpolated(cudaImageListf& images, float x,float y, int img, bool &outside)
	{
		return images.interpolate(x,y,img, outside);
	}

	// Assumes pixel is always within image bounds
	static __device__ float Index(cudaImageListf& images, int x,int y, int img)
	{
		return images.pixel(x, y, img);
	}
};

texture<float, cudaTextureType2D, cudaReadModeElementType> qi_image_texture_linear(0, cudaFilterModeLinear); // Un-normalized

// Using the lower-accuracy interpolation of texture hardware
class ImageSampler_SimpleTextureRead {
public:
	static void BindTexture(cudaImageListf& images) { images.bind(qi_image_texture_linear); }
	static void UnbindTexture(cudaImageListf& images) { images.unbind(qi_image_texture_linear);  }

	// All interpolated texture/images fetches go through here
	static __device__ float Interpolated(cudaImageListf& images, float x,float y, int img, bool &outside)
	{
		if (x < 0 || x >= images.w-1 || y < 0 || y >= images.h-1) {
			outside=true;
			return 0.0f;
		} else  {
			outside=false;
			return tex2D(qi_image_texture_linear, ofs(x),ofs(y) + img*images.h);
		}
	}

	// Assumes pixel is always within image bounds
	static __device__ float Index(cudaImageListf& images, int x,int y, int img)
	{
		return tex2D(qi_image_texture_linear, ofs(x),ofs(y) + img*images.h);
	}

private:
	static __device__ float ofs(float x) { return x+0.5f; }
};

// According to this, textures bindings can be switched after the asynchronous kernel is launched
// https://devtalk.nvidia.com/default/topic/392245/texture-binding-and-stream/
texture<float, cudaTextureType2D, cudaReadModeElementType> qi_image_texture_nearest(0, cudaFilterModePoint); // Un-normalized

// Using 4 texture fetches + standard 32 bit interpolation
class ImageSampler_InterpolatedTexture {
public:
	static void BindTexture(cudaImageListf& images) { images.bind(qi_image_texture_nearest); }
	static void UnbindTexture(cudaImageListf& images) { images.unbind(qi_image_texture_nearest);  }

	// All interpolated texture/images fetches go through here
	static __device__ float Interpolated(cudaImageListf& images, float x,float y, int img, bool &outside)
	{
		return images.interpolateFromTexture (qi_image_texture_nearest, x, y, img, outside);
	}

	// Assumes pixel is always within image bounds
	static __device__ float Index(cudaImageListf& images, int x,int y, int img)
	{
		return tex2D(qi_image_texture_nearest, x+0.5f,y+0.5f + img*images.h);
	}
};


typedef ImageSampler_InterpolatedTexture ImageSampler_Tex;
//typedef ImageSampler_SimpleTextureRead ImageSampler_Tex;
