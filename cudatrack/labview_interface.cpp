#include "nivision.h"
#include "extcode.h"
#include "niimaq.h"

#include <Windows.h>
#include "utils.h"
#include "GPUBase.h"

#include "tracker.h"

#define CALLCONV _FUNCC

static std::string lastErrorMsg;


DLL_EXPORT Tracker* CALLCONV create_tracker(int w,int h)
{
	return new Tracker(w,h);
}

DLL_EXPORT void CALLCONV free_tracker(Tracker* tracker)
{
	if (tracker->isValid()) {
		delete tracker;
	}
}


DLL_EXPORT Image* CALLCONV copy_image(Image* image)
{
	ImageInfo info, dstInfo;
	imaqGetImageInfo(image, &info);

	Image* output = imaqCreateImage(IMAQ_IMAGE_U8, 0);
	imaqSetImageSize(output, info.xRes, info.yRes);
	imaqGetImageInfo(output, &dstInfo);

	uint8_t *src = (uint8_t*)imaqGetPixelAddress(image, Point());
	for (int y=0;y<info.yRes;y++) {
		src += info.pixelsPerLine;

		for (int x=0;x<info.xRes;x++) {
			Point pt = {x,y};
			PixelValue v;
			v.grayscale = 255-src[x];
			imaqSetPixel(output, pt, v);
		}
	}

	return output;
}

DLL_EXPORT void CALLCONV compute_com(Tracker* tracker, float* pos)
{
	vector2f com = tracker->ComputeCOM();
	pos[0] = com.x;
	pos[1] = com.y;
}




DLL_EXPORT void CALLCONV testMsg() 
{
	MessageBox(0, "hi", "Msg:", MB_OK);
}




DLL_EXPORT void CALLCONV xcor_localize(Tracker* tracker, const vector2f& initial, vector2f& position)
{

}

DLL_EXPORT void CALLCONV generate_test_image(Tracker* tracker, float xpos, float ypos, float S)
{
	tracker->loadTestImage(xpos, ypos, S);
}

DLL_EXPORT void CALLCONV get_current_image(Tracker* tracker, Image* target)
{
	uchar *data = new uchar[tracker->width*tracker->height];
	tracker->copyToHost(data, tracker->width*sizeof(uchar));

	int width, height;
	imaqGetImageSize(target, &width, &height);
	if (width != tracker->width || height != tracker->height)
		imaqSetImageSize(target, tracker->width, tracker->height);

	imaqArrayToImage(target, data, tracker->width, tracker->height);
	delete[] data;
}

DLL_EXPORT void CALLCONV set_image(Tracker* tracker, Image* img)
{
	ImageInfo info;
	imaqGetImageInfo(img, &info);
	if (info.imageType == IMAQ_IMAGE_U8)
		tracker->setImage( (uchar*)info.imageStart, info.pixelsPerLine);
}
