#include <cstdio>
#include "jpeglib.h"
#include "utils.h"
#include "TeLibJpeg\jmemdstsrc.h"


struct my_error_mgr {
  struct jpeg_error_mgr pub;	/* "public" fields */
};

int ReadJPEGFile(uchar* srcbuf, int srclen, uchar** data, int* width, int*height)
{
  struct jpeg_decompress_struct cinfo;

  JSAMPARRAY buffer;		/* Output row buffer */
  int row_stride;		/* physical row width in output buffer */
  my_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr.pub);
  jpeg_create_decompress(&cinfo);

  j_mem_src(&cinfo, srcbuf, srclen);

  /* Step 3: read file parameters with jpeg_read_header() */
  jpeg_read_header(&cinfo, TRUE);

  jpeg_start_decompress(&cinfo);
  row_stride = cinfo.output_width * cinfo.output_components;
  /* Make a one-row-high sample array that will go away when done with image */
  buffer = (*cinfo.mem->alloc_sarray) ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

  *width = cinfo.output_width;
  *height = cinfo.output_height;
  *data = new uchar[cinfo.output_width*cinfo.output_height];

//  ResizeLVArray2D(output, cinfo.output_height, cinfo.output_width);

  /* Step 6: while (scan lines remain to be read) */
  /*           jpeg_read_scanlines(...); */

  /* Here we use the library's state variable cinfo.output_scanline as the
   * loop counter, so that we don't have to keep track ourselves.
   */
  uchar* dst = *data;
  while (cinfo.output_scanline < cinfo.output_height) {
    /* jpeg_read_scanlines expects an array of pointers to scanlines.
     * Here the array is only one element long, but you could ask for
     * more than one scanline at a time if that's more convenient.
     */
    jpeg_read_scanlines(&cinfo, buffer, 1);
	/* Assume put_scanline_someplace wants a pointer and sample count. */

	unsigned char* src = buffer[0];
	if (cinfo.output_components == 1) {
		memcpy(dst, src, cinfo.output_width);
	} else {
		for (int x=0;x<cinfo.output_width;x++)
			dst[x] = src[x * cinfo.output_components];
	}
	dst += cinfo.output_width;
  }

  /* Step 7: Finish decompression */
  jpeg_finish_decompress(&cinfo);
  /* We can ignore the return value since suspension is not possible
   * with the stdio data source.
   */

  /* Step 8: Release JPEG decompression object */

  /* This is an important step since it will release a good deal of memory. */
  jpeg_destroy_decompress(&cinfo);

  /* After finish_decompress, we can close the input file.
   * Here we postpone it until after no more JPEG errors are possible,
   * so as to simplify the setjmp error logic above.  (Actually, I don't
   * think that jpeg_destroy can do an error exit, but why assume anything...)
   */
//  fclose(infile);

  /* At this point you may want to check to see whether any corrupt-data
   * warnings occurred (test whether jerr.pub.num_warnings is nonzero).
   */

  return 1;
}

