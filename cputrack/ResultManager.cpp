#include "std_incl.h"
#include "ResultManager.h"
#include "utils.h"

ResultManager::ResultManager(const char *outfile, const char* frameInfoFile, ResultManagerConfig *cfg)
{
	config = *cfg;
	outputFile = outfile;
	this->frameInfoFile = frameInfoFile;

	startFrame = 0;
	lastSaveFrame = 0;
	processedFrames = 0;
	capturedFrames = 0;
	localizationsDone = 0;

	qtrk = 0;

	thread = Threads::Create(ThreadLoop, this);
	quit=false;

	remove(outfile);
	remove(frameInfoFile);

	dbgprintf("Allocating ResultManager with %d beads and writeinterval %d\n", cfg->numBeads, cfg->writeInterval);
}

ResultManager::~ResultManager()
{
	quit = true;
	Threads::WaitAndClose(thread);

	DeleteAllElems(frameResults);
}


void ResultManager::StoreResult(LocalizationResult *r)
{
	int index = r->job.frame - startFrame;

	if (index >= frameResults.size()) {
		dbgprintf("dropping result. Result provided for unregistered frame %d. Current frames registered: %d\n", 
			r->job.frame, startFrame + frameResults.size());
		return; // add errors?
	}

	LocalizationResult scaled = *r;
	// Make roi-centered pos
	scaled.pos = scaled.pos - vector3f( qtrk->cfg.width*0.5f, qtrk->cfg.height*0.5f, 0);
	scaled.pos = ( scaled.pos + config.offset ) * config.scaling;
	FrameResult* fr = frameResults[index];
	fr->results[r->job.zlutIndex] = scaled;
	fr->count++;

	// Advance fullFrames
	frameCountMutex.lock();
	while (processedFrames - startFrame < frameResults.size() && frameResults[processedFrames-startFrame]->count == config.numBeads)
		processedFrames ++;
	localizationsDone ++;
	frameCountMutex.unlock();
}

void ResultManager::Write()
{
	FILE* f = outputFile.empty () ? 0 : fopen(outputFile.c_str(), "a");
	FILE* finfo = frameInfoFile.empty() ? 0 : fopen(frameInfoFile.c_str(), "a");
	
	resultMutex.lock();
	if (config.binaryOutput) {
		for (uint j=lastSaveFrame; j<processedFrames;j++)
		{
			auto fr = frameResults[j-startFrame];
			if (f) {
				fwrite(&j, sizeof(uint), 1, f);
				fwrite(&fr->timestamp, sizeof(double), 1, f);
				for (int i=0;i<config.numBeads;i++) 
				{
					LocalizationResult *r = &fr->results[i];
					fwrite(&r->pos, sizeof(vector3f), 1, f);
				}
			}
			if (finfo)
				fwrite(&fr->timestamp, sizeof(double), 1, finfo);
		}
	}
	else {
		for (uint k=lastSaveFrame; k<processedFrames;k++)
		{
			auto fr = frameResults[k-startFrame];
			if (f) {
				fprintf(f,"%d\t%f\t", k, fr->timestamp);
				for (int i=0;i<config.numBeads;i++) 
				{
					LocalizationResult *r = &fr->results[i];
					fprintf(f, "%.5f\t%.5f\t%.5f\t", r->pos.x,r->pos.y,r->pos.z);
				}
				fputs("\n", f);
			}
			if (finfo) {
				fprintf(finfo,"%d\t%f\t", k, fr->timestamp);
				for (int i=0;i<config.numFrameInfoColumns;i++)
					fprintf(finfo, "%.5f\t", fr->frameInfo[i]);
				fputs("\n", finfo);
			}
		}
	}

	dbgprintf("Saved frame %d to %d\n", lastSaveFrame, processedFrames);

	if(f) fclose(f);
	if(finfo) fclose(finfo);
	frameCountMutex.lock();
	lastSaveFrame = processedFrames;
	frameCountMutex.unlock();

	resultMutex.unlock();
}


void ResultManager::SetTracker(QueuedTracker *qtrk)
{
	trackerMutex.lock();
	this->qtrk = qtrk;
	trackerMutex.unlock();
}

bool ResultManager::Update()
{
	trackerMutex.lock();

	if (!qtrk) {
		trackerMutex.unlock();
		return 0;
	}

	const int NResultBuf = 40;
	LocalizationResult resultbuf[NResultBuf];

	int count = qtrk->PollFinished( resultbuf, NResultBuf );

	resultMutex.lock();
	for (int i=0;i<count;i++)
		StoreResult(&resultbuf[i]);
	resultMutex.unlock();

	trackerMutex.unlock();

	if (processedFrames - lastSaveFrame >= config.writeInterval) {
		Write();
	}

	if (config.maxFramesInMemory>0 && frameResults.size () > config.maxFramesInMemory) {
		int del = frameResults.size()-config.maxFramesInMemory;
		dbgprintf("Removing %d frames from memory\n", del);
		
		for (int i=0;i<del;i++)
			delete frameResults[i];
		frameResults.erase(frameResults.begin(), frameResults.begin()+del);

		frameCountMutex.lock();
		startFrame += del;
		frameCountMutex.unlock();
	}

	return count>0;
}

void ResultManager::ThreadLoop(void *param)
{
	ResultManager* rm = (ResultManager*)param;

	while(true) {
		if (!rm->Update())
			Threads::Sleep(20);

		if (rm->quit)
			break;
	}
}

int ResultManager::GetBeadPositions(int startFrame, int endFrame, int bead, LocalizationResult* results)
{
	int start = startFrame - this->startFrame;

	if (endFrame > processedFrames)
		endFrame = processedFrames;

	int end = endFrame - this->startFrame;

	if (start < 0)
		return 0;

	resultMutex.lock();
	for (int i=start;i<end;i++){
		results[i-start] = frameResults[i]->results[bead];
	}
	resultMutex.unlock();

	return end-start;
}


void ResultManager::Flush()
{
	Write();

	resultMutex.lock();

	// Dump stats about unfinished frames for debugging
	for (int i=0;i<frameResults.size();i++) {
		FrameResult *fr = frameResults[i];
		dbgprintf("Frame %d. TS: %f, Count: %d\n", i, fr->timestamp, fr->count);
		if (fr->count != config.numBeads) {
			for (int j=0;j<fr->results.size();j++) {
				if( fr->results[j].job.locType == 0 )
					dbgprintf("%d, ", j );
			}
			dbgprintf("\n");
		}
	}

	resultMutex.unlock();
}


void ResultManager::GetFrameCounters(int* startFrame, int *processedFrames, int *lastSaveFrame, int *capturedFrames, int *localizationsDone)
{
	frameCountMutex.lock();
	if (startFrame) *startFrame = this->startFrame;
	if (processedFrames) *processedFrames = this->processedFrames;
	if (lastSaveFrame) *lastSaveFrame = this->lastSaveFrame;
	if (localizationsDone) *localizationsDone = this->localizationsDone;
	frameCountMutex.unlock();

	if (capturedFrames) {
		resultMutex.lock();
		*capturedFrames = this->capturedFrames;
		resultMutex.unlock();
	}
}

int ResultManager::GetResults(LocalizationResult* results, int startFrame, int numFrames)
{
	frameCountMutex.lock();

	if (startFrame >= this->startFrame && numFrames+startFrame <= processedFrames)  {
		resultMutex.lock();
		for (int f=0;f<numFrames;f++) {
			int index = f + startFrame - this->startFrame;
			for (int j=0;j<config.numBeads;j++)
				results[config.numBeads*f+j] = frameResults[index]->results[j];
		}

		resultMutex.unlock();
	}
	frameCountMutex.unlock();

	return numFrames;
}


int ResultManager::StoreFrameInfo(double timestamp, float* columns)
{
	resultMutex.lock();
	auto fr = new FrameResult( config.numBeads, config.numFrameInfoColumns);
	fr->timestamp = timestamp;
	for(int i=0;i<config.numFrameInfoColumns;i++)
		fr->frameInfo[i]=columns[i];
	frameResults.push_back (fr);
	int nfr = ++capturedFrames;
	resultMutex.unlock();
	return nfr;
}


int ResultManager::GetFrameCount()
{
	resultMutex.lock();
	int nfr = capturedFrames;
	resultMutex.unlock();
	return nfr;
}

