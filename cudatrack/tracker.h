#pragma once


#define TRACKER_MAGIC 0xf843e49a

#pragma pack(push, 4)
struct vector2f {
	float x,y;
};
#pragma pack(pop)

typedef unsigned int uint;
typedef unsigned char uchar;

// This also needs to correspond to the LabView code creating the image!
typedef float pixel_t;

// Stores all buffer variables needed for the tracker. 
class TrackerBuffer;

class Tracker
{
public:
	uint magic; // magic ID, to recognise invalid objects passed from labview
	uint width, height;

	TrackerBuffer* buffer;

	Tracker(uint w,uint h);
	~Tracker();
	bool isValid() { return magic==TRACKER_MAGIC; }

	vector2f XCorLocalize(vector2f initial);
	vector2f ComputeCOM();
	void setImage(pixel_t* image, uint pitchInBytes);
	void loadTestImage(float xpos, float ypos, float S);
	void copyToHost(pixel_t* data, uint pitchInBytes);
	void* getCurrentBufferImage();
};

