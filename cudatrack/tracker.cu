
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "tracker.h"
#include "Array2D.h"


template<typename T> void safeCudaFree(T*& ptr) {
	if (ptr) {
		cudaFree(ptr);
		ptr = 0;
	}
}

class TrackerBuffer
{
public:
	Array2D<pixel_t, float>* image;
	reducer_buffer<float> reduceBuffer;

	TrackerBuffer(uint w,uint h) : reduceBuffer(w,h) {
		image = 0;
	}
	~TrackerBuffer()
	{
		if (image) delete image;
	}
};

template<typename T, typename TC>
static vector2f ComputeCOM(Array2D<T, TC>* image, reducer_buffer<TC>& reduceBuffer)
{
	vector2f com;
	com.x = image->momentX(reduceBuffer);
	com.y = image->momentY(reduceBuffer);
	float sum = image->sum(reduceBuffer);
	com.x /= sum;
	com.y /= sum;
	return com;
}

Tracker::Tracker(uint w, uint h) {
	magic = TRACKER_MAGIC;

	width = w;
	height = h;
	buffer = new TrackerBuffer(w,h);
}

Tracker::~Tracker() {
}

void Tracker::setImage(pixel_t* data, uint pitchInBytes) {
	
}


struct TestImgComputePixel {
	float xpos, ypos, S;
	float compute(uint x, uint y) {
/*		if (x==0&&y==0)
			printf("value: %f", value);
*/
		float X = x + 0.5f - xpos;
		float Y = y + 0.5f - ypos;
		float r = sqrtf(X*X+Y*Y)+1;
		float v = sinf( (r-10)*2*3.141593f*S);
		return v*v / (r * r * S);
	}
};


void Tracker::loadTestImage(float xpos, float ypos, float S)
{
	if (!buffer->image) {
		buffer->image = new Array2D<pixel_t,float>(width, height);
	}
	TestImgComputePixel pixel_op = { xpos, ypos, 1.0f/S };

	// generate
	float* buf = new float[width*height];
	for (uint y=0;y<height;y++)
		for(uint x=0;x<width;x++)
			buf[y*width+x] = pixel_op.compute(x,y);

	// normalize
	float minv, maxv;
	minv=maxv=buf[0];
	for (int k=0;k<width*height;k++) {
		minv=std::min(minv, buf[k]);
		maxv=std::max(maxv, buf[k]);
	}
	// convert to uchar
	uchar *ibuf = new uchar[width*height];
	for (int k=0;k<width*height;k++)
		ibuf[k]= 255.0f * (buf[k]-minv)/(maxv-minv);
	delete[] buf;

	buffer->image->set(ibuf, sizeof(pixel_t)*width);
	delete[] ibuf;
}

vector2f Tracker::ComputeCOM()
{
	if (!buffer->image)
		return vector2f();

	return ::ComputeCOM(buffer->image, buffer->reduceBuffer);
}

vector2f Tracker::XCorLocalize(vector2f initial)
{
	vector2f estimate;

	return initial;
}

void Tracker::copyToHost(pixel_t* data, uint pitchInBytes)
{
	if (buffer->image)
		buffer->image->copyToHost(data, pitchInBytes);
}

void* Tracker::getCurrentBufferImage() {
	return buffer->image;
}
