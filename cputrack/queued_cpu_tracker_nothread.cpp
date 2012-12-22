
#include "queued_cpu_tracker_nothread.h"


QueuedTracker* CreateQueuedTracker(QTrkSettings* s) {
	return new QueuedCPUTracker(s);
}

static int PDT_BytesPerPixel(QTRK_PixelDataType pdt) {
	const int pdtBytes[] = {1, 2, 4};
	return pdtBytes[(int)pdt];
}

int QueuedCPUTracker::GetResultCount()
{
	return resultCount;
}

int QueuedCPUTracker::GetJobCount()
{
	return 0;
}

QueuedCPUTracker::QueuedCPUTracker(QTrkSettings* pcfg)
{
	cfg = *pcfg;
	resultCount = 0;

	zluts = 0;
	zlut_count = zlut_planes = zlut_res = 0;

	tracker = new CPUTracker(cfg.width, cfg.height, cfg.xc1_profileLength);
}

QueuedCPUTracker::~QueuedCPUTracker()
{
	delete tracker;
}


void QueuedCPUTracker::Start()
{
}

void QueuedCPUTracker::ClearResults()
{
	results.clear();
}

void QueuedCPUTracker::ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, 
				LocalizeType locType, uint id, vector3f* initial, uint zlutIndex, uint zlutPlane)
{
	if (pdt == QTrkU8) {
		tracker->SetImage8Bit(data, cfg.width);
	} else if (pdt == QTrkU16) {
		tracker->SetImage16Bit((ushort*)data, cfg.width*2);
	} else {
		tracker->SetImageFloat((float*)data);
	}

	LocalizationResult result={};
	result.id = id;
	result.locType = locType;
	result.zlutIndex = zlutIndex;

	vector2f com = tracker->ComputeBgCorrectedCOM();

	bool boundaryHit = false;

	switch((LocalizeType)(locType&Localize2DMask)) {
	case LocalizeXCor1D:
		result.firstGuess = com;
		result.pos = tracker->ComputeXCorInterpolated(com, cfg.xc1_iterations, cfg.xc1_profileWidth, boundaryHit);
		break;
	case LocalizeOnlyCOM:
		result.firstGuess = result.pos = com;
		break;
	case LocalizeQI:
		result.firstGuess = com;
		result.pos = tracker->ComputeQI(com, cfg.qi_iterations, cfg.qi_radialsteps, cfg.qi_angularsteps, cfg.qi_minradius, cfg.qi_maxradius, boundaryHit);
		break;
	}

	result.error = boundaryHit ? 1 : 0;

	if(locType & LocalizeZ) {
		result.z = tracker->ComputeZ(result.pos, cfg.zlut_angularsteps, zlutIndex);
	} else if (locType & LocalizeBuildZLUT) {
		float* zlut = GetZLUTByIndex(zlutIndex);
		tracker->ComputeRadialProfile(&zlut[zlutPlane * zlut_res], zlut_res, cfg.zlut_angularsteps, cfg.zlut_minradius, cfg.zlut_maxradius, result.pos, &boundaryHit);
	}

	results.push_back(result);
	resultCount++;
}

void QueuedCPUTracker::SetZLUT(float* data, int num_zluts, int planes, int res)
{
	if (zluts) delete[] zluts;
	zluts = new float[planes*res*num_zluts];
	std::fill(zluts,zluts+(planes*res*num_zluts), 0.0f);
	zlut_planes = planes;
	zlut_res = res;
	zlut_count = num_zluts;
	if(data)
		std::copy(data, data+(planes*res*num_zluts), zluts);

	tracker->SetZLUT(zluts, zlut_planes, zlut_res, zlut_count, cfg.zlut_minradius, cfg.zlut_maxradius, cfg.zlut_angularsteps, false, false);
}


float* QueuedCPUTracker::GetZLUT(int *count, int* planes,int *res)
{
	float* cp = new float [zlut_planes*zlut_res*zlut_count];
	std::copy(zluts, zluts+(zlut_planes*zlut_res*zlut_count), cp);

	if (count) *count = zlut_count;
	if (planes) *planes = zlut_planes;
	if (res) *res = zlut_res;

	return cp;
}


int QueuedCPUTracker::PollFinished(LocalizationResult* dstResults, int maxResults)
{
	int numResults = 0;
	while (numResults < maxResults && !results.empty()) {
		dstResults[numResults++] = results.front();
		results.pop_front();
		resultCount--;
	}
	return numResults;
}


void QueuedCPUTracker::GenerateTestImage(float* dst, float xp,float yp, float z, float photoncount)
{
	ImageData img(dst,cfg.width,cfg.height);
	::GenerateTestImage(img,xp,yp,z,photoncount);
}


