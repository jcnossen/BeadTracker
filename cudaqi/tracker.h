#pragma once

#include <stdint.h>

#define TRACKER_MAGIC 0xf843e49a

#pragma pack(push, 4)
struct vector2f {
	float x,y;
};
#pragma pack(pop)

typedef uint32_t uint;


class Tracker
{
public:
	uint32_t magic; // magic ID, to recognise invalid objects passed from labview
	float* d_buf;
	uint8_t* d_original;
	uint32_t width, height;

	Tracker(uint w,uint h);
	~Tracker();
	bool isValid() { return magic==TRACKER_MAGIC; }

	vector2f XCorLocalize(vector2f initial);
	vector2f ComputeCOM();
	void setImage(uint8_t* image);
};

