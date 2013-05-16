
#pragma once

#include "QueuedTracker.h"
#include <list>
#include "threads.h"

// Runs a seperate result fetching thread
class ResultManager
{
public:
	ResultManager(QueuedTracker *qtrk, int numBeads);
	~ResultManager();

	void EnableFetcher(bool enable);
	void SetOutputFile(const char *filename);
	void SetWriteInterval(int nFrames);

	int GetLastFrame();
	void GetPositions(int frame, LocalizationResult *results); // results[nBeads]

protected:
	void Write();

	class Block
	{
	public:
		LocalizationResult* results;
	};

	std::list<Block> blocks;

	QueuedTracker* qtrk;
	std::string outputFile;
	Threads::Handle* thread;	
};
