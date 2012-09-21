#pragma once


#define TRACKER_MAGIC 0xf843e49a

#pragma pack(push, 4)
struct vector2f {
	float x,y;
};
#pragma pack(pop)

typedef unsigned int uint;
typedef unsigned char uchar;

template<typename T>
class Array2D;

class Tracker
{
public:
	uint magic; // magic ID, to recognise invalid objects passed from labview
	uint width, height;
	Array2D<uchar>* original;
	Array2D<float>* sampleBuffer;

	Tracker(uint w,uint h);
	~Tracker();
	bool isValid() { return magic==TRACKER_MAGIC; }

	vector2f XCorLocalize(vector2f initial);
	vector2f ComputeCOM();
	void setImage(uchar* image);
};

