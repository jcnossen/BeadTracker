#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "std_incl.h"

#include "QueuedCUDATracker.h"

QueuedTracker* CreateQueuedTracker(QTrkSettings* cfg)
{
	return new QueuedCUDATracker(cfg);
}

QueuedCUDATracker::QueuedCUDATracker(QTrkSettings *cfg)
{
	this->cfg = *cfg;
}

QueuedCUDATracker::~QueuedCUDATracker()
{
}

void QueuedCUDATracker::Start() 
{

}



void QueuedCUDATracker::ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint id, uint zlutIndex)
{
}


int QueuedCUDATracker::PollFinished(LocalizationResult* results, int maxResult)
{
	return 0;
}

void QueuedCUDATracker::SetZLUT(float* data, int planes, int res, int numLUTs)
{
}

void QueuedCUDATracker::ComputeRadialProfile(float *image, int width, int height, float* dst, int profileLength, vector2f center)
{
}



int QueuedCUDATracker::GetJobCount()
{
	return 0;
}

int QueuedCUDATracker::GetResultCount()
{
	return 0;
}


void QueuedCUDATracker::GenerateTestImage(float* dst, float xp,float yp, float z, float photoncount)
{
}

