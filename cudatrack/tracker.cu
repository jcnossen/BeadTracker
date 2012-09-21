
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "tracker.h"

template<typename T> void safeCudaFree(T*& ptr) {
	if (ptr) {
		cudaFree(ptr);
		ptr = 0;
	}
}

Tracker::Tracker(uint w, uint h) {
	magic = TRACKER_MAGIC;
}

Tracker::~Tracker() {
}

void Tracker::setImage(uchar* data) {

}

