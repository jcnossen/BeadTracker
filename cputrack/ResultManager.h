
#pragma once

#include "QueuedTracker.h"
#include <list>
#include "threads.h"

// Labview interface packing
#pragma pack(push,1)
struct ResultManagerConfig
{
	int numBeads;
	vector3f scaling;
	vector3f offset; // output will be   (position + offset) * scaling
	int writeInterval;
	uint maxFramesInMemory; // 0 for infinite
	uint8_t binaryOutput;
};
#pragma pack(pop)

// Runs a seperate result fetching thread
class ResultManager
{
public:
	ResultManager(const char *outfile, ResultManagerConfig *cfg);
	~ResultManager();

	void EnableResultFetch(QueuedTracker *qtrk);

	//int GetStartFrame() { return startFrame; }
	//int GetLastFrame() { return fullFrames; }
	//int GetLastWrittenFrame() { return lastSaveFrame; }
	int GetBeadPositions(int startFrame, int endFrame, int bead, LocalizationResult* r);
	int GetResults(LocalizationResult* results, int startFrame, int numResults);
	void Flush();

	void GetFrameCounters(int* startFrame, int *fullFrames, int *lastSaveFrame);

	// Configure output of an additional frame info file
	void SetFrameInfoFile(const char* outfile, int numcols);
	void SetFrameInfo(int frame, float* columns);

protected:
	void Write();
	void StoreResult(LocalizationResult* r);

	struct FrameResult
	{
		FrameResult() {count=0; results=0; }
		LocalizationResult* results;
		int count;
	};

	Threads::Mutex frameCountMutex, resultMutex;

	std::deque< FrameResult > frameResults;
	int startFrame; // startFrame for frameResults
	int fullFrames; // frame where all data is retrieved (all beads)
	int lastSaveFrame;
	ResultManagerConfig config;

	QueuedTracker* qtrk;
	std::string outputFile;
	Threads::Handle* thread;
	bool quit;

	static void ThreadLoop(void *param);
	bool Update();
};
