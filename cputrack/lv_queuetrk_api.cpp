/*
Labview API for the functionality in QueuedTracker.h
*/

#include "utils.h"
#include "labview.h"
#include "QueuedTracker.h"

#include "jpeglib.h"
#include "TeLibJpeg\jmemdstsrc.h"


CDLL_EXPORT void DLL_CALLCONV qtrk_set_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pZlut)
{
	LVArray3D<float>* zlut = *pZlut;

	int numLUTs = zlut->dimSizes[0];
	int planes = zlut->dimSizes[1];
	int res = zlut->dimSizes[2];

	dbgprintf("Setting ZLUT size: %d beads, %d planes, %d radialsteps\n", numLUTs, planes, res);
	
	tracker->SetZLUT(zlut->elem, planes, res, numLUTs);
}

CDLL_EXPORT void DLL_CALLCONV qtrk_get_ZLUT(QueuedTracker* tracker, LVArray3D<float>** pzlut)
{
	int dims[3];

	float* zlut = tracker->GetZLUT(&dims[0], &dims[1], &dims[2]);
	ResizeLVArray3D(pzlut, dims[0], dims[1], dims[2]);
	std::copy(zlut, zlut+(*pzlut)->numElem(), (*pzlut)->elem);
	delete[] zlut;
}

CDLL_EXPORT QueuedTracker* qtrk_create(QTrkSettings* settings, int startNow)
{
	QueuedTracker* tracker = CreateQueuedTracker(settings);
	if (startNow)
		tracker->Start();
	return tracker;
}

CDLL_EXPORT void qtrk_start(QueuedTracker* qtrk)
{
	qtrk->Start();
}


CDLL_EXPORT void qtrk_destroy(QueuedTracker* qtrk)
{
	delete qtrk;
}


CDLL_EXPORT void qtrk_queue_u16(QueuedTracker* qtrk, LVArray2D<ushort>** data, uint locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane)
{
	qtrk->ScheduleLocalization( (uchar*)(*data)->elem, sizeof(ushort)*(*data)->dimSizes[1], QTrkU16, (LocalizeType)locType, id, initialPos, zlutIndex, zlutPlane);
}

CDLL_EXPORT void qtrk_queue_u8(QueuedTracker* qtrk, LVArray2D<uchar>** data, uint locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane)
{
	qtrk->ScheduleLocalization( (*data)->elem, sizeof(uchar)*(*data)->dimSizes[1], QTrkU8, (LocalizeType) locType, id, initialPos, zlutIndex, zlutPlane);
}

CDLL_EXPORT void qtrk_queue_float(QueuedTracker* qtrk, LVArray2D<float>** data, uint locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane)
{
	qtrk->ScheduleLocalization( (uchar*) (*data)->elem, sizeof(float)*(*data)->dimSizes[1], QTrkFloat, (LocalizeType) locType, id, initialPos, zlutIndex, zlutPlane);
}

CDLL_EXPORT void test_array_passing(LVArray2D<float>** data, float* data2, int* len)
{
	int total=len[0]*len[1];
	for(int i=0;i<total;i++)
		dbgprintf("[%d] Data=%f, Data2=%f\n", i,(*data)->elem[i], data2[i]);
}

CDLL_EXPORT void qtrk_queue(QueuedTracker* qtrk, uchar* data, int pitch, uint pdt, uint locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane)
{
	qtrk->ScheduleLocalization(data, pitch, (QTRK_PixelDataType)pdt, (LocalizeType) locType, id, initialPos, zlutIndex, zlutPlane);
}

CDLL_EXPORT void qtrk_queue_array(QueuedTracker* qtrk, LVArray2D<uchar>** data, uint pdt, uint locType, uint id, vector3f* initialPos, uint zlutIndex, uint zlutPlane)
{
	uint pitch;

	if (pdt == QTrkFloat) 
		pitch = sizeof(float);
	else if(pdt == QTrkU16) 
		pitch = 2;
	else pitch = 1;

	pitch *= (*data)->dimSizes[1]; // LVArray2D<uchar> type works for ushort and float as well
	dbgprintf("zlutindex: %d, zlutplane: %d\n", zlutIndex,zlutPlane);
	qtrk_queue(qtrk, (*data)->elem, pitch, pdt, locType, id, initialPos, zlutIndex, zlutPlane);
}

CDLL_EXPORT void qtrk_clear_results(QueuedTracker* qtrk)
{
	qtrk->ClearResults();
}


CDLL_EXPORT int qtrk_jobcount(QueuedTracker* qtrk)
{
	return qtrk->GetJobCount();
}


CDLL_EXPORT int qtrk_resultcount(QueuedTracker* qtrk)
{
	return qtrk->GetResultCount();
}

static bool compareResultsByID(const LocalizationResult& a, const LocalizationResult& b) {
	return a.id<b.id;
}

CDLL_EXPORT int qtrk_get_results(QueuedTracker* qtrk, LocalizationResult* results, int maxResults, int sortByID)
{
	int resultCount = qtrk->PollFinished(results, maxResults);

	if (sortByID) {
		std::sort(results, results+resultCount, compareResultsByID);
	}

	return resultCount;
}

CDLL_EXPORT void DLL_CALLCONV qtrk_generate_test_image(QueuedTracker* tracker, LVArray2D<ushort>** image, float xp, float yp, float size, float photoncount)
{
	int w=tracker->cfg.width, h =tracker->cfg.height;
	ResizeLVArray2D(image, h,w);
	
	float *d = new float[w*h];
	tracker->GenerateTestImage(d, xp, yp, size, photoncount );
	floatToNormalizedInt((*image)->elem, d, w,h, (ushort)((1<<16)-1));
	delete[] d;
}

CDLL_EXPORT void DLL_CALLCONV qtrk_generate_image_from_lut(LVArray2D<float>** image, LVArray2D<float>** lut, float LUTradius, vector2f* position, float z, float M, float photonCountPP)
{
	ImageData img((*image)->elem, (*image)->dimSizes[1], (*image)->dimSizes[0]);
	ImageData zlut((*lut)->elem, (*lut)->dimSizes[1], (*lut)->dimSizes[0]);

	GenerateImageFromLUT(&img, &zlut, LUTradius, *position, z, M);
	img.normalize();
	if(photonCountPP>0)
		ApplyPoissonNoise(img, photonCountPP);
}




struct my_error_mgr {
  struct jpeg_error_mgr pub;	/* "public" fields */
};


CDLL_EXPORT int DLL_CALLCONV qtrk_read_jpeg_from_file(const char* filename, LVArray2D<uchar>** dstImage)
{
	int w,h;

	FILE *f = fopen(filename, "rb");

	fseek(f, 0, SEEK_END);
	int len = ftell(f);
	fseek(f, 0, SEEK_SET);

	uchar* buf=new uchar[len];
	fread(buf, 1,len, f);
	fclose(f);

  struct jpeg_decompress_struct cinfo;

  JSAMPARRAY buffer;		/* Output row buffer */
  int row_stride;		/* physical row width in output buffer */
  my_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr.pub);
  jpeg_create_decompress(&cinfo);

  j_mem_src(&cinfo, buf, len);

  /* Step 3: read file parameters with jpeg_read_header() */
  jpeg_read_header(&cinfo, TRUE);
  jpeg_start_decompress(&cinfo);
  
  if (cinfo.output_components != 1) {
	  delete[] buf;
	  return 0;
  }

  w = cinfo.output_width;
  h = cinfo.output_height;
	if ( (*dstImage)->dimSizes[0] != h || (*dstImage)->dimSizes[1] != w )
		ResizeLVArray2D(dstImage, h, w);

  row_stride = cinfo.output_width * cinfo.output_components;
  /* Make a one-row-high sample array that will go away when done with image */
 // buffer = (*cinfo.mem->alloc_sarray) ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

  while (cinfo.output_scanline < cinfo.output_height) {
	uchar* jpeg_buf = & (*dstImage)->elem [cinfo.output_scanline * w];
    jpeg_read_scanlines(&cinfo, &jpeg_buf, 1);
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  delete[] buf;
  return 1;
}



