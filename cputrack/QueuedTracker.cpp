#include "std_incl.h"
#include "QueuedTracker.h"
#include "AsyncScheduler.h"

QueuedTracker::QueuedTracker()
{
	asyncScheduler = 0;
}

QueuedTracker::~QueuedTracker()
{
	if (asyncScheduler)
		delete asyncScheduler;
}

void QueuedTracker::ScheduleFrameAsync(uchar *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo)
{
	if (!asyncScheduler) {
		asyncScheduler = new AsyncScheduler(this);
	}

	asyncScheduler->Schedule(imgptr, pitch, width, height, positions, numROI, pdt, jobInfo);
}

bool QueuedTracker::IsAsyncScheduleDone(uchar* ptr)
{
	return asyncScheduler->IsFinished(ptr);
}


bool QueuedTracker::IsAsyncSchedulerIdle()
{
	return asyncScheduler ? asyncScheduler->IsEmpty() : true;
}

void QueuedTracker::ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, LocalizeType locType, uint frame, uint timestamp, vector3f* initial, uint zlutIndex, uint zlutPlane)
{
	LocalizationJob j;
	j.frame= frame;
	j.locType = locType;
	j.timestamp = timestamp;
	if (initial) j.initialPos = *initial;
	j.zlutIndex = zlutIndex;
	j.zlutPlane = zlutPlane;
	ScheduleLocalization(data,pitch,pdt,&j);
}


