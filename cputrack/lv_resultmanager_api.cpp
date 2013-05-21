/// --------------------------------------------------------------------------------------
// ResultManager

#include "std_incl.h"
#include "labview.h"
#include "ResultManager.h"

CDLL_EXPORT ResultManager* DLL_CALLCONV rm_create(QueuedTracker* qtrk, const char *file, ResultManagerConfig* cfg)
{
	ResultManager* rm = new ResultManager(qtrk, file, cfg);
	return rm;
}

CDLL_EXPORT void DLL_CALLCONV rm_destroy(ResultManager* rm)
{
	delete rm;
}


CDLL_EXPORT void DLL_CALLCONV rm_getbeadresults(ResultManager* rm, int start, int end, int bead, LocalizationResult* results)
{
	rm->GetBeadPositions(start,end,bead,results);
}


CDLL_EXPORT void DLL_CALLCONV rm_getframecounters(ResultManager* rm, int* startFrame, int* lastSaveFrame, int* endFrame)
{
	rm->GetFrameCounters(startFrame, endFrame, lastSaveFrame);
}

CDLL_EXPORT void DLL_CALLCONV rm_flush(ResultManager* rm)
{
	rm->Flush();
}

CDLL_EXPORT void DLL_CALLCONV rm_getresults(ResultManager* rm, LocalizationResult* results, int startFrame, int numFrames)
{
	rm->GetResults(results, startFrame, numFrames);
}
