#include "std_incl.h"
#include "ResultManager.h"
#include "utils.h"

ResultManager::ResultManager(const char *outfile, ResultManagerConfig *cfg)
{
	config = *cfg;
	outputFile = outfile;

	startFrame = 0;
	lastSaveFrame = 0;
	fullFrames = 0;

	qtrk = 0;

	thread = Threads::Create(ThreadLoop, this);
	quit=false;

	frameResults.resize(100);
	for (int i=0;i<frameResults.size();i++) 
		frameResults[i].results =new LocalizationResult[config.numBeads];

	remove(outfile);
}

ResultManager::~ResultManager()
{
	quit = true;
	Threads::WaitAndClose(thread);

	for(int i=0;i<frameResults.size();i++)
		delete[] frameResults[i].results;
}

void ResultManager::StoreResult(LocalizationResult *r)
{
	int index = r->job.frame - startFrame;

	if (index >= frameResults.size()) {
		int prevsize = frameResults.size();
		frameResults.resize(frameResults.size()*2);

		for (int i=prevsize;i<frameResults.size();i++) 
			frameResults[i].results =new LocalizationResult[config.numBeads];
	}

	LocalizationResult scaled = *r;
	// Make roi-centered pos
	scaled.pos = scaled.pos - vector3f( qtrk->cfg.width*0.5f, qtrk->cfg.height*0.5f, 0);
	scaled.pos = ( scaled.pos + config.offset ) * config.scaling;
	FrameResult& fr = frameResults[index];
	fr.results[r->job.zlutIndex] = scaled;
	fr.count++;

	// Advance fullFrames
	frameCountMutex.lock();
	while (fullFrames - startFrame < frameResults.size() && frameResults[fullFrames-startFrame].count == config.numBeads)
		fullFrames ++;
	frameCountMutex.unlock();
}

void ResultManager::Write()
{
	FILE* f = fopen(outputFile.c_str(), "a");
	
	resultMutex.lock();
	if (config.binaryOutput) {
		for (int fr=lastSaveFrame; fr<fullFrames;fr++)
		{
			auto results = frameResults[fr-startFrame].results;
			fwrite(&results[0].job.frame, sizeof(uint), 1, f);
			fwrite(&results[0].job.timestamp, sizeof(uint), 1, f);
			for (int i=0;i<config.numBeads;i++) 
			{
				LocalizationResult *r = &frameResults[fr-startFrame].results[i];
				fwrite(&r->pos, sizeof(vector3f), 1, f);
			}
		}
	}
	else {
		for (int fr=lastSaveFrame; fr<fullFrames;fr++)
		{
			auto results = frameResults[fr-startFrame].results;
			fprintf(f,"%d\t%d\t",results[0].job.frame, results[0].job.timestamp);
			for (int i=0;i<config.numBeads;i++) 
			{
				LocalizationResult *r = &results[i];
				fprintf(f, "%.5f\t%.5f\t%.5f\t", r->pos.x,r->pos.y,r->pos.z);
			}

			fputs("\n", f);
		}
	}

	dbgprintf("Saved frame %d to %d\n", lastSaveFrame, fullFrames);

	fclose(f);
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
			delete[] frameResults[i].results;
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
		results[i-start] = frameResults[i].results[bead];
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
				results[config.numBeads*f+j] = frameResults[index].results[j];
		}

		resultMutex.unlock();
	}
	frameCountMutex.unlock();

	return numFrames;
}

