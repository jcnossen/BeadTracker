/// --------------------------------------------------------------------------------------
// ResultManager

#include "std_incl.h"
#include "labview.h"
#include "ResultManager.h"

CDLL_EXPORT ResultManager* DLL_CALLCONV rm_create(const char *file, ResultManagerConfig* cfg)
{
	ResultManager* rm = new ResultManager(file, cfg);
	return rm;
}

CDLL_EXPORT void DLL_CALLCONV rm_set_tracker(ResultManager* rm, QueuedTracker* qtrk)
{
	rm->EnableResultFetch(qtrk);
}

CDLL_EXPORT void DLL_CALLCONV rm_destroy(ResultManager* rm)
{
	delete rm;
}


CDLL_EXPORT int DLL_CALLCONV rm_getbeadresults(ResultManager* rm, int start, int end, int bead, LocalizationResult* results)
{
	return rm->GetBeadPositions(start,end,bead,results);
}


CDLL_EXPORT void DLL_CALLCONV rm_getframecounters(ResultManager* rm, int* startFrame, int* lastSaveFrame, int* endFrame)
{
	rm->GetFrameCounters(startFrame, endFrame, lastSaveFrame);
}

CDLL_EXPORT void DLL_CALLCONV rm_flush(ResultManager* rm)
{
	rm->Flush();
}

CDLL_EXPORT int DLL_CALLCONV rm_getresults(ResultManager* rm, LocalizationResult* results, int startFrame, int numFrames)
{
	return rm->GetResults(results, startFrame, numFrames);
}
