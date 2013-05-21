
#pragma once

#include "QueuedTracker.h"
#include <list>
#include "threads.h"

// Runs a seperate result fetching thread
class ResultManager
{
public:
	ResultManager(QueuedTracker *qtrk, int numBeads, const char* outfile);
	~ResultManager();

	void EnableFetcher(bool enable);
	void SetWriteInterval(int nFrames);

	int GetLastFrame() { return fullFrames; }
	void GetBeadPositions(int startFrame, int endFrame, int bead, LocalizationResult* r);


protected:
	void Write();
	void StoreResult(LocalizationResult* r);

	struct FrameResult
	{
		FrameResult() {count=0; results=0; }
		LocalizationResult* results;
		int count;
	};

	std::deque< FrameResult > frameResults;
	int startFrame; // startFrame for frameResults
	int fullFrames; // frame where all data is retrieved (all beads)

	QueuedTracker* qtrk;
	std::string outputFile;
	Threads::Handle* thread;
	int numBeads;
	bool quit;

	static void ThreadLoop(void *param);
};
