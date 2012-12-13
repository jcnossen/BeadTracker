
#include "queued_cpu_tracker.h"

QueuedTracker* CreateQueuedTracker(QTrkSettings* s) {
	return new QueuedCPUTracker(s);
}

static int PDT_BytesPerPixel(QTRK_PixelDataType pdt) {
	const int pdtBytes[] = {1, 2, 4};
	return pdtBytes[(int)pdt];
}

int QueuedCPUTracker::GetResultCount()
{
	results_mutex.lock();
	int rc = resultCount;
	results_mutex.unlock();
	return rc;
}


void QueuedCPUTracker::JobFinished(QueuedCPUTracker::Job* j)
{
	jobs_buffer_mutex.lock();
	jobs_buffer.push_back(j);
	jobs_buffer_mutex.unlock();
}

QueuedCPUTracker::Job* QueuedCPUTracker::GetNextJob()
{
	QueuedCPUTracker::Job *j = 0;
	jobs_mutex.lock();
	if (!jobs.empty()) {
		j = jobs.front();
		jobs.pop_front();
		jobCount --;
	}
	jobs_mutex.unlock();
	return j;
}

QueuedCPUTracker::Job* QueuedCPUTracker::AllocateJob()
{
	QueuedCPUTracker::Job *j;
	jobs_buffer_mutex.lock();
	if (!jobs_buffer.empty()) {
		j = jobs_buffer.back();
		jobs_buffer.pop_back();
	} else 
		j = new Job;
	jobs_buffer_mutex.unlock();
	return j;
}

void QueuedCPUTracker::AddJob(Job* j)
{
	jobs_mutex.lock();
	jobs.push_back(j);
	jobCount++;
	jobs_mutex.unlock();
}

int QueuedCPUTracker::GetJobCount()
{
	int jc;
	jobs_mutex.lock();
	jc = jobCount;
	jobs_mutex.unlock();
	return jc;
}

QueuedCPUTracker::QueuedCPUTracker(QTrkSettings* pcfg)
{
	cfg = *pcfg;
	quitWork = false;

	if (cfg.numThreads < 0) {
		// preferably 
		#ifdef WIN32	
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		cfg.numThreads = sysInfo.dwNumberOfProcessors;
		#else
		cfg.numThreads = 4;
		#endif
		dbgprintf("Using %d threads\n", cfg.numThreads);
	}

	jobCount = 0;
	resultCount = 0;

	zluts = 0;
	zlut_count = zlut_planes = zlut_res = 0;
}

QueuedCPUTracker::~QueuedCPUTracker()
{
	// wait for threads to finish
	quitWork = true;

	for (int k=0;k<threads.size();k++) {
		Threads::WaitAndClose(threads[k].thread);
		delete threads[k].tracker;
	}

	// free job memory
	DeleteAllElems(jobs);
	DeleteAllElems(jobs_buffer);
}


void QueuedCPUTracker::Start()
{
	quitWork = false;
	threads.resize(cfg.numThreads);
	for (int k=0;k<cfg.numThreads;k++) {
		threads[k].tracker = new CPUTracker(cfg.width, cfg.height, cfg.xc1_profileLength);
		threads[k].manager = this;
	}

	for (int k=0;k<threads.size();k++) {
		threads[k].thread = Threads::Create(WorkerThreadMain, &threads[k]);
	}
}

DWORD QueuedCPUTracker::WorkerThreadMain(void* arg)
{
	Thread* th = (Thread*)arg;
	QueuedCPUTracker* this_ = th->manager;

	while (!this_->quitWork) {
		Job* j = this_->GetNextJob();
		if (j) {
			this_->ProcessJob(th, j);
			this_->JobFinished(j);
		} else {
			#ifdef WIN32
				Sleep(1);
			#endif
		}
	}
	dbgprintf("Thread %p ending.\n", arg);
	return 0;
}

void QueuedCPUTracker::ProcessJob(Thread* th, Job* j)
{
	if (j->dataType == QTrkU8) {
		th->tracker->SetImage8Bit(j->data, cfg.width);
	} else if (j->dataType == QTrkU16) {
		th->tracker->SetImage16Bit((ushort*)j->data, cfg.width*2);
	} else {
		th->tracker->SetImageFloat((float*)j->data);
	}

	LocalizationResult result={};
	result.id = j->id;
	result.locType = j->locType;
	result.zlutIndex = j->zlut;

	vector2f com = th->tracker->ComputeBgCorrectedCOM();

	LocalizeType locType = (LocalizeType)(j->locType&Localize2DMask);
	bool boundaryHit = false;

	switch(locType) {
	case LocalizeXCor1D:
		result.firstGuess = com;
		result.pos = th->tracker->ComputeXCorInterpolated(com, cfg.xc1_iterations, cfg.xc1_profileWidth, boundaryHit);
		break;
	case LocalizeOnlyCOM:
		result.firstGuess = result.pos = com;
		break;
	case LocalizeQI:
		result.firstGuess = com;
		result.pos = th->tracker->ComputeQI(com, cfg.qi_iterations, cfg.qi_radialsteps, cfg.qi_angularsteps, cfg.qi_minradius, cfg.qi_maxradius, boundaryHit);
		break;
	}

	if(j->locType & LocalizeZ) {
		result.z = th->tracker->ComputeZ(result.pos, cfg.zlut_angularsteps, j->zlut, &boundaryHit);
	}
	result.error = boundaryHit ? 1 : 0;

	results_mutex.lock();
	results.push_back(result);
	resultCount++;
	results_mutex.unlock();
}

void QueuedCPUTracker::SetZLUT(float* data, int planes, int res, int num_zluts)
{
	if (zluts) delete[] zluts;
	zluts = new float[planes*res*num_zluts];
	zlut_planes = planes;
	zlut_res = res;
	zlut_count = num_zluts;
}

void QueuedCPUTracker::ComputeRadialProfile(float *image, int width, int height, float* dst, int profileLen, vector2f center)
{
	ImageData imgData (image,  width,height);
	::ComputeRadialProfile(dst, profileLen, cfg.zlut_angularsteps, cfg.zlut_minradius, cfg.zlut_maxradius, center, &imgData, 0, imgData.mean());
}


void QueuedCPUTracker::ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, 
				LocalizeType locType, uint id, vector3f* initialPos, uint zlutIndex)
{
	Job* j = AllocateJob();
	int dstPitch = PDT_BytesPerPixel(pdt) * cfg.width;

	if(!j->data || j->dataType != pdt) {
		if (j->data) delete[] j->data;
		j->data = new uchar[dstPitch * cfg.height];
	}
	for (int y=0;y<cfg.height;y++)
		memcpy(&j->data[dstPitch*y], &data[pitch*y], dstPitch);

	j->dataType = pdt;
	j->id = id;
	j->locType = locType;
	j->zlut = zlutIndex;
	if(initialPos) 
		j->initialPos = *initialPos;
	
	AddJob(j);
}

int QueuedCPUTracker::PollFinished(LocalizationResult* dstResults, int maxResults)
{
	int numResults = 0;
	results_mutex.lock();
	while (numResults < maxResults && !results.empty()) {
		dstResults[numResults++] = results.front();
		results.pop_front();
		resultCount--;
	}
	results_mutex.unlock();
	return numResults;
}


void QueuedCPUTracker::GenerateTestImage(float* dst, float xp,float yp, float z, float photoncount)
{
	ImageData img(dst,cfg.width,cfg.height);
	::GenerateTestImage(img,xp,yp,z,photoncount);
}


