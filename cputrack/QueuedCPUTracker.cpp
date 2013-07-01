#include "std_incl.h"
#include "QueuedCPUTracker.h"
#include <float.h>

#ifndef CUDA_TRACK
QueuedTracker* CreateQueuedTracker(const QTrkComputedConfig& cc){ 
	return new QueuedCPUTracker(cc);
}
void SetCUDADevices(int* dev, int ndev) {
}
#endif

static int PDT_BytesPerPixel(QTRK_PixelDataType pdt) {
	const int pdtBytes[] = {1, 2, 4};
	return pdtBytes[(int)pdt];
}

bool QueuedCPUTracker::IsIdle()
{
	return GetQueueLength() == 0;
}

int QueuedCPUTracker::GetResultCount()
{
	results_mutex.lock();
	int rc = resultCount;
	results_mutex.unlock();
	return rc;
}

void QueuedCPUTracker::ClearResults()
{
	results_mutex.lock();
	resultCount = 0;
	results.clear();
	results_mutex.unlock();
}

void QueuedCPUTracker::JobFinished(QueuedCPUTracker::Job* j)
{
	jobs_buffer_mutex.lock();
	jobs_buffer.push_back(j);
	jobs_buffer_mutex.unlock();

	jobs_mutex.lock();
	jobsInProgress--;
	jobs_mutex.unlock();
}

QueuedCPUTracker::Job* QueuedCPUTracker::GetNextJob()
{
	QueuedCPUTracker::Job *j = 0;
	jobs_mutex.lock();
	if (!jobs.empty()) {
		j = jobs.front();
		jobs.pop_front();
		jobCount --;
		jobsInProgress ++;
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

int QueuedCPUTracker::GetQueueLength(int *maxQueueLength)
{
	int jc;
	jobs_mutex.lock();
	jc = jobCount + jobsInProgress;
	jobs_mutex.unlock();

	if (maxQueueLength) 
		*maxQueueLength = this->maxQueueSize;

	return jc;
}

QueuedCPUTracker::QueuedCPUTracker(const QTrkComputedConfig& cc) 
	: jobs_mutex("jobs"), jobs_buffer_mutex("jobs_buffer"), results_mutex("results")
{
	cfg = cc;
	quitWork = false;

	if (cfg.numThreads < 0) {
		cfg.numThreads = Threads::GetCPUCount();
		dbgprintf("Using %d threads\n", cfg.numThreads);
	} 

	if (cfg.numThreads == 0) {
		noThreadTracker = new CPUTracker(cfg.width, cfg.height, cfg.xc1_profileLength);
	} else
		noThreadTracker = 0;

	maxQueueSize = std::max(2,cfg.numThreads) * 50;
	jobCount = 0;
	resultCount = 0;

	zluts = 0;
	zlut_count = zlut_planes = 0;
	processJobs = false;
	jobsInProgress = 0;

	Start();
}

QueuedCPUTracker::~QueuedCPUTracker()
{
	// wait for threads to finish
	quitWork = true;

	for (int k=0;k<threads.size();k++) {
		Threads::WaitAndClose(threads[k].thread);
		delete threads[k].tracker;
	}

	if (noThreadTracker)
		delete noThreadTracker;

	// free job memory
	DeleteAllElems(jobs);
	DeleteAllElems(jobs_buffer);

	if (zluts) {
		delete[] zluts;
	}
}

void QueuedCPUTracker::Break(bool brk)
{
	processJobs = !brk;
}


void QueuedCPUTracker::Start()
{
	quitWork = false;
	if (noThreadTracker) 
		return;

	threads.resize(cfg.numThreads);
	for (int k=0;k<cfg.numThreads;k++) {
		threads[k].tracker = new CPUTracker(cfg.width, cfg.height, cfg.xc1_profileLength);
		threads[k].manager = this;
	}

	for (int k=0;k<threads.size();k++) {
		threads[k].thread = Threads::Create(WorkerThreadMain, &threads[k]);
		Threads::SetBackgroundPriority(threads[k].thread, true);
	}

	processJobs = true;
}

void QueuedCPUTracker::WorkerThreadMain(void* arg)
{
	Thread* th = (Thread*)arg;
	QueuedCPUTracker* this_ = th->manager;

	while (!this_->quitWork) {
		Job* j = 0;
		if (this_->processJobs) 
			j = this_->GetNextJob();

		if (j) {
			this_->ProcessJob(th->tracker, j);
			this_->JobFinished(j);
		} else {
			Threads::Sleep(1);
		}
	}
	dbgprintf("Thread %p ending.\n", arg);
}

void QueuedCPUTracker::ProcessJob(CPUTracker* trk, Job* j)
{
	if (j->dataType == QTrkU8) {
		trk->SetImage8Bit(j->data, cfg.width);
	} else if (j->dataType == QTrkU16) {
		trk->SetImage16Bit((ushort*)j->data, cfg.width*2);
	} else {
		trk->SetImageFloat((float*)j->data);
	}

//	dbgprintf("Job: id %d, bead %d\n", j->id, j->zlut);

	LocalizationResult result={};
	result.job = j->job;

	vector2f com = trk->ComputeBgCorrectedCOM(cfg.com_bgcorrection);

	if (_isnan(com.x) || _isnan(com.y))
		com = vector2f(cfg.width/2,cfg.height/2);

	LocalizeType locType = (LocalizeType)(j->job.locType&Localize2DMask);
	bool boundaryHit = false;

	switch(locType) {
	case LocalizeXCor1D: {
		result.firstGuess = com;
		vector2f resultPos = trk->ComputeXCorInterpolated(com, cfg.xc1_iterations, cfg.xc1_profileWidth, boundaryHit);
		result.pos.x = resultPos.x;
		result.pos.y = resultPos.y;
		break;}
	case LocalizeOnlyCOM:
		result.firstGuess.x = result.pos.x = com.x;
		result.firstGuess.y = result.pos.y = com.y;
		break;
	case LocalizeQI:{
		result.firstGuess = com;
		vector2f resultPos = trk->ComputeQI(com, cfg.qi_iterations, cfg.qi_radialsteps, cfg.qi_angstepspq, cfg.qi_angstep_factor, cfg.qi_minradius, cfg.qi_maxradius, boundaryHit);
		result.pos.x = resultPos.x;
		result.pos.y = resultPos.y;
		break;}
	case LocalizeGaussian2D:{
		result.firstGuess = com;
		vector2f xy = trk->Compute2DGaussianMLE(com, cfg.gauss2D_iterations);
		result.pos = vector3f(xy.x,xy.y,0.0f);
		break;}
	}

	if(j->job.LocType() & LocalizeZ) {
		result.pos.z = trk->ComputeZ(result.pos2D(), cfg.zlut_angularsteps, j->job.zlutIndex, false, &boundaryHit);
	} else if (j->job.LocType() & LocalizeBuildZLUT) {
		float* zlut = GetZLUTByIndex(j->job.zlutIndex);
		trk->ComputeRadialProfile(&zlut[j->job.zlutPlane * cfg.zlut_radialsteps], cfg.zlut_radialsteps, cfg.zlut_angularsteps, cfg.zlut_minradius, cfg.zlut_maxradius, result.pos2D(), false, &boundaryHit);
	}

#ifdef _DEBUG
	dbgprintf("pos[%d]: x=%f, y=%f, z=%f\n", result.job.zlutIndex, result.pos.x, result.pos.y, result.pos.z);
#endif

	result.error = boundaryHit ? 1 : 0;

	results_mutex.lock();
	results.push_back(result);
	resultCount++;
	results_mutex.unlock();
}

void QueuedCPUTracker::SetZLUT(float* data, int num_zluts, int planes, float* zcmp)
{
//	jobs_mutex.lock();
//	results_mutex.lock();

	if (zluts) delete[] zluts;
	int res = cfg.zlut_radialsteps;
	int total = num_zluts*res*planes;
	if (total > 0) {
		zluts = new float[planes*res*num_zluts];
		std::fill(zluts,zluts+(planes*res*num_zluts), 0.0f);
		zlut_planes = planes;
		zlut_count = num_zluts;
		if(data)
			std::copy(data, data+(planes*res*num_zluts), zluts);
	}
	else
		zluts = 0;

	if (zcmp) {
		this->zcmp.assign(zcmp, zcmp+res);
	}
	else
		this->zcmp.clear();

	UpdateZLUTs();
//	results_mutex.unlock();
//	jobs_mutex.unlock();
}

void QueuedCPUTracker::UpdateZLUTs()
{
	for (int i=0;i<threads.size();i++){
		threads[i].tracker->SetZLUT(zluts, zlut_planes, cfg.zlut_radialsteps, zlut_count, cfg.zlut_minradius, cfg.zlut_maxradius, cfg.zlut_angularsteps, false, false, zcmp.empty() ? 0 : &zcmp[0]);
	}
}



float* QueuedCPUTracker::GetZLUT(int *count, int* planes)
{
	float* cp = 0;
	if (zlut_planes*cfg.zlut_radialsteps*zlut_count>0) {
		results_mutex.lock();
		cp = new float [zlut_planes*cfg.zlut_radialsteps*zlut_count];
		std::copy(zluts, zluts+(zlut_planes*cfg.zlut_radialsteps*zlut_count), cp);
		results_mutex.unlock();
	}

	if(count) *count = zlut_count;
	if(planes) *planes = zlut_planes;

	return cp;
}

void QueuedCPUTracker::ScheduleLocalization(uchar* data, int pitch, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo)
{
	if (processJobs) {
		while(maxQueueSize != 0 && GetQueueLength () >= maxQueueSize)
			Threads::Sleep(5);
	}

	Job* j = AllocateJob();
	int dstPitch = PDT_BytesPerPixel(pdt) * cfg.width;

	if(!j->data || j->dataType != pdt) {
		if (j->data) delete[] j->data;
		j->data = new uchar[dstPitch * cfg.height];
	}
	for (int y=0; y<cfg.height; y++)
		memcpy(&j->data[dstPitch*y], &data[pitch*y], dstPitch);

	j->dataType = pdt;
	j->job = *jobInfo;
	if (!zluts || jobInfo->zlutIndex < 0 || jobInfo->zlutIndex>=this->zlut_count || 
		( (jobInfo->locType&LocalizeBuildZLUT) && ( jobInfo->zlutPlane < 0 || jobInfo->zlutPlane >= this->zlut_planes) ))
	{
		j->job.locType &= ~(LocalizeBuildZLUT|LocalizeZ);
	}

#ifdef _DEBUG
	dbgprintf("Scheduled job: frame %d, bead %d\n", j->job.zlutPlane, j->job.zlutIndex);
#endif

	AddJob(j);

	if (noThreadTracker) {
		Job* j_ = GetNextJob();
		ProcessJob(noThreadTracker, j_);
		JobFinished(j_);
	}
}

int QueuedCPUTracker::PollFinished(LocalizationResult* dstResults, int maxResults)
{
	int numResults = 0;
	results_mutex.lock();
	while (numResults < maxResults && !results.empty()) {
		dstResults[numResults++] = results.back();
		results.pop_back();
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


void QueuedCPUTracker::ScheduleFrame(uchar *imgptr, int pitch, int width, int height, ROIPosition *positions, int numROI, QTRK_PixelDataType pdt, const LocalizationJob *jobInfo)
{
	uchar* img = (uchar*)imgptr;
	int bpp = PDT_BytesPerPixel(pdt);
	for (int i=0;i<numROI;i++){
		ROIPosition& pos = positions[i];

		if (pos.x < 0 || pos.y < 0 || pos.x + cfg.width > width || pos.y + cfg.height > height)
			continue;

		uchar *roiptr = &img[pitch * pos.y + pos.x * bpp];
		LocalizationJob job = *jobInfo;
		job.zlutIndex = i + jobInfo->zlutIndex; // used as offset
		ScheduleLocalization(roiptr, pitch, pdt, &job);
	}
}
