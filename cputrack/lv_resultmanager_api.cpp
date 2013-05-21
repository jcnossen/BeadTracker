/// --------------------------------------------------------------------------------------
// ResultManager

#include "std_incl.h"
#include "labview.h"
#include "ResultManager.h"

CDLL_EXPORT ResultManager* DLL_CALLCONV rm_create(QueuedTracker* qtrk, const char *file, int numBeads)
{
}

CDLL_EXPORT void DLL_CALLCONV rm_destroy(ResultManager* rm)
{
	delete rm;
}


CDLL_EXPORT void DLL_CALLCONV rm_getbeadresults(ResultManager* rm, int start, int end, int bead, LocalizationResult* results)
{
	rm->GetBeadPositions(start,end,bead,results);
}
