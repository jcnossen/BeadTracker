#include "std_incl.h"
#include "QueuedTracker.h"
#include "AsyncScheduler.h"
#include "utils.h"

void QTrkComputedConfig::Update()
{
	int roi = width / 2;

	zlut_maxradius = roi*zlut_roi_coverage;
	float zlut_perimeter = 2*3.141593f*zlut_maxradius;
	zlut_angularsteps = zlut_perimeter*zlut_angular_coverage;
	zlut_radialsteps = (zlut_maxradius-zlut_minradius)*zlut_radial_coverage;

	qi_maxradius = roi*qi_roi_coverage;
	float qi_perimeter = 2*3.141593f*qi_maxradius;
	qi_angstepspq = qi_perimeter*qi_angular_coverage/4;
	qi_radialsteps = (qi_maxradius-qi_minradius)*qi_radial_coverage;

	qi_radialsteps = NearestPowerOf2(qi_radialsteps);
}

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


