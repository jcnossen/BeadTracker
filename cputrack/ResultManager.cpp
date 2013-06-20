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
	fullFrames = 0;

	qtrk = 0;

	thread = Threads::Create(ThreadLoop, this);
	quit=false;

	frameResults.resize(100);
	for (int i=0;i<frameResults.size();i++) 
		frameResults[i] =new FrameResult(config.numBeads, config.numFrameInfoColumns);

	remove(outfile);
	remove(frameInfoFile);
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
	while (fullFrames - startFrame < frameResults.size() && frameResults[fullFrames-startFrame]->count == config.numBeads)
		fullFrames ++;
	frameCountMutex.unlock();
}

void ResultManager::Write()
{
	FILE* f = fopen(outputFile.c_str(), "a");
	FILE* finfo = fopen(frameInfoFile.c_str(), "a");
	
	resultMutex.lock();
	if (config.binaryOutput) {
		for (uint j=lastSaveFrame; j<fullFrames;j++)
		{
			auto fr = frameResults[j-startFrame];
			fwrite(&j, sizeof(uint), 1, f);
			fwrite(&fr->timestamp, sizeof(double), 1, f);
			fwrite(&fr->timestamp, sizeof(double), 1, finfo);
			for (int i=0;i<config.numBeads;i++) 
			{
				LocalizationResult *r = &fr->results[i];
				fwrite(&r->pos, sizeof(vector3f), 1, f);
			}
		}
	}
	else {
		for (uint k=lastSaveFrame; k<fullFrames;k++)
		{
			auto fr = frameResults[k-startFrame];
			fprintf(f,"%d\t%f\t", k, fr->timestamp);
			fprintf(finfo,"%d\t%f\t", k, fr->timestamp);
			for (int i=0;i<config.numFrameInfoColumns;i++)
				fprintf(f, "%.5f\t", fr->frameInfo[i]);
			fputs("\n", finfo);
			for (int i=0;i<config.numBeads;i++) 
			{
				LocalizationResult *r = &fr->results[i];
				fprintf(f, "%.5f\t%.5f\t%.5f\t", r->pos.x,r->pos.y,r->pos.z);
			}
			fputs("\n", f);
		}
	}

	dbgprintf("Saved frame %d to %d\n", lastSaveFrame, fullFrames);

	fclose(f);
	fclose(finfo);
	frameCountMutex.lock();
	lastSaveFrame = fullFrames;
	frameCountMutex.unlock();

	resultMutex.unlock();
}


void ResultManager::EnableResultFetch(QueuedTracker *qtrk)
{
	this->qtrk = qtrk;
}

bool ResultManager::Update()
{
	if (!qtrk)
		return 0;

	const int NResultBuf = 10;
	LocalizationResult resultbuf[NResultBuf];

	int count = qtrk->PollFinished( resultbuf, NResultBuf );

	resultMutex.lock();
	for (int i=0;i<count;i++)
		StoreResult(&resultbuf[i]);
	resultMutex.unlock();

	if (fullFrames - lastSaveFrame >= config.writeInterval) {
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

	if (endFrame > fullFrames)
		endFrame = fullFrames;

	int end = endFrame - this->startFrame;
	
	for (int i=start;i<end;i++){
		results[i-start] = frameResults[i]->results[bead];
	}

	return end-start;
}


void ResultManager::Flush()
{
	Write();
}


void ResultManager::GetFrameCounters(int* startFrame, int *fullFrames, int *lastSaveFrame)
{
	frameCountMutex.lock();
	if (startFrame) *startFrame = this->startFrame;
	if (fullFrames) *fullFrames = this->fullFrames;
	if (lastSaveFrame) *lastSaveFrame = this->lastSaveFrame;
	frameCountMutex.unlock();
}

int ResultManager::GetResults(LocalizationResult* results, int startFrame, int numFrames)
{
	frameCountMutex.lock();

	if (startFrame >= this->startFrame && numFrames+startFrame <= fullFrames)  {
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


void ResultManager::StoreFrameInfo(double timestamp, float* columns)
{
	resultMutex.lock();
	auto fr = new FrameResult( config.numBeads, config.numFrameInfoColumns);
	fr->timestamp = timestamp;
	for(int i=0;i<config.numFrameInfoColumns;i++)
		fr->frameInfo[i]=columns[i];
	frameResults.push_back (fr);
	resultMutex.unlock();
}

