/// --------------------------------------------------------------------------------------
// ResultManager

#include "std_incl.h"
#include "labview.h"
#include "ResultManager.h"
#include "hash_templates.h"
#include "utils.h"

static qtrk::hash_set <ResultManager*> rm_instances;

static bool ValidRM(ResultManager* rm, ErrorCluster* err)
{
	if(rm_instances.find(rm) == rm_instances.end()) {
		ArgumentErrorMsg(err, "Invalid ResultManager instance passed.");
		return false;
	}
	return true;
}

CDLL_EXPORT void DLL_CALLCONV rm_destroy_all()
{
	DeleteAllElems(rm_instances);
}

CDLL_EXPORT ResultManager* DLL_CALLCONV rm_create(const char *file, const char *frameinfo, ResultManagerConfig* cfg)
{
	ResultManager* rm = new ResultManager(file, frameinfo, cfg);
	rm_instances.insert(rm);
	return rm;
}

CDLL_EXPORT void DLL_CALLCONV rm_set_tracker(ResultManager* rm, QueuedTracker* qtrk, ErrorCluster* err)
{
	if (ValidRM(rm, err)) {
		rm->SetTracker(qtrk);
	}
}

CDLL_EXPORT void DLL_CALLCONV rm_destroy(ResultManager* rm, ErrorCluster  *err)
{
	if (ValidRM(rm, err)) {
		rm_instances.erase(rm);
		delete rm;
	}
}

CDLL_EXPORT int DLL_CALLCONV rm_store_frame_info(ResultManager* rm, double timestamp, float* cols, ErrorCluster* err)
{
	if (ValidRM(rm, err)) {
		return rm->StoreFrameInfo(timestamp, cols);
	} 
	return 0;
}

CDLL_EXPORT int DLL_CALLCONV rm_getbeadresults(ResultManager* rm, int start, int numFrames, int bead, LocalizationResult* results, ErrorCluster* err)
{
	if (ValidRM(rm, err)) {
		if (bead < 0 || bead >= rm->Config().numBeads)
			ArgumentErrorMsg(err,SPrintf( "Invalid bead index: %d. Accepted range: [0-%d]", bead, rm->Config().numBeads));
		else
			return rm->GetBeadPositions(start,start+numFrames,bead,results);
	}
	return 0;
}


CDLL_EXPORT void DLL_CALLCONV rm_getframecounters(ResultManager* rm, int* startFrame, int* lastSaveFrame, int* endFrame, int *capturedFrames, ErrorCluster* err)
{
	if (ValidRM(rm, err)) {
		rm->GetFrameCounters(startFrame, endFrame, lastSaveFrame, capturedFrames);
	}
}

CDLL_EXPORT void DLL_CALLCONV rm_flush(ResultManager* rm, ErrorCluster* err)
{
	if (ValidRM(rm, err)) {
		rm->Flush();
	}
}

CDLL_EXPORT int DLL_CALLCONV rm_getresults(ResultManager* rm, int startFrame, int numFrames, LocalizationResult* results, ErrorCluster* err)
{
	if (ValidRM(rm,err)) {
		return rm->GetResults(results, startFrame, numFrames);
	}
	return 0;
}
