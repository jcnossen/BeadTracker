
#pragma once

#include "QueuedTracker.h"
#include <list>
#include "threads.h"

// Labview interface packing
#pragma pack(push,1)
struct ResultManagerConfig
{
	int numBeads, numFrameInfoColumns;
	vector3f scaling;
	vector3f offset; // output will be (position + offset) * scaling
	int writeInterval; // [frames]
	uint maxFramesInMemory; // 0 for infinite
	uint8_t binaryOutput;
};
#pragma pack(pop)

// Runs a seperate result fetching thread
class ResultManager
{
public:
	ResultManager(const char *outfile, const char *frameinfo, ResultManagerConfig *cfg);
	~ResultManager();

	void SaveSection(int start, int end, const char *beadposfile, const char *infofile);
	void SetTracker(QueuedTracker *qtrk);

	int GetBeadPositions(int startFrame, int endFrame, int bead, LocalizationResult* r);
	int GetResults(LocalizationResult* results, int startFrame, int numResults);
	void Flush();

	void GetFrameCounters(int* startFrame=0, int *processedFrames=0, int *lastSaveFrame=0, int *capturedFrames=0, int *localizationsDone=0);
	int StoreFrameInfo(double timestamp, float* columns); // return #frames
	int GetFrameCount();

	const ResultManagerConfig& Config() { return config; }

protected:
	void Write();
	void StoreResult(LocalizationResult* r);

	struct FrameResult
	{
		FrameResult(int nResult, int nFrameInfo) : frameInfo(nFrameInfo), results(nResult) { count=0; timestamp=0;}
		std::vector<LocalizationResult> results;
		std::vector<float> frameInfo;
		int count;
		double timestamp;
	};

	Threads::Mutex frameCountMutex, resultMutex, trackerMutex;

	std::deque< FrameResult* > frameResults;
	int startFrame; // startFrame for frameResults
	int processedFrames; // frame where all data is retrieved (all beads)
	int lastSaveFrame;
	int capturedFrames;  // lock by resultMutex
	int localizationsDone;
	ResultManagerConfig config;

	QueuedTracker* qtrk;

	std::string outputFile, frameInfoFile;
	Threads::Handle* thread;
	bool quit;

	static void ThreadLoop(void *param);
	bool Update();
};
