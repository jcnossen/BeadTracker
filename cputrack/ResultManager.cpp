#include "std_incl.h"
#include "ResultManager.h"
#include "utils.h"

ResultManager::ResultManager(QueuedTracker *qtrk, int numBeads, const char *outfile)
{
	this->numBeads=numBeads;
	outputFile = outfile;

	startFrame = 0;

	thread = Threads::Create(ThreadLoop, this);
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
		int prevsize = frameResults.size();
		frameResults.resize(frameResults.size()*2);

		for (int i=prevsize;i<frameResults.size();i++) 
			frameResults[i].results =new LocalizationResult[numBeads];
	}

	FrameResult& fr = frameResults[index];
	fr.results[r->job.zlutIndex] = *r;
	fr.count++;

	// Advance fullFrames
	while (fullFrames - startFrame < frameResults.size() && frameResults[fullFrames-startFrame].count == numBeads)
		fullFrames ++;
}



void ResultManager::ThreadLoop(void *param)
{
	ResultManager* rm = (ResultManager*)param;
	const int NResultBuf = 10;
	LocalizationResult resultbuf[NResultBuf];

	while(true) {
		int count = rm->qtrk->PollFinished( resultbuf, NResultBuf );
		
		if (count > 0) {
			for (int i=0;i<count;i++)
				rm->StoreResult(&resultbuf[i]);
		} else
			Threads::Sleep(20);

		if (rm->quit)
			break;
	}
}

