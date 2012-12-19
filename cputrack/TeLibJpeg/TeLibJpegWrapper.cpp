/************************************************************************************
TerraLib - a library for developing GIS applications.
Copyright ¨ 2001-2004 INPE and Tecgraf/PUC-Rio.

This code is part of the TerraLib library.
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

You should have received a copy of the GNU Lesser General Public
License along with this library.

The authors reassure the license terms regarding the warranties.
They specifically disclaim any warranties, including, but not limited to,
the implied warranties of merchantability and fitness for a particular purpose.
The library provided hereunder is on an "as is" basis, and the authors have no
obligation to provide maintenance, support, updates, enhancements, or modifications.
In no event shall INPE and Tecgraf / PUC-Rio be held liable to any party for direct,
indirect, special, incidental, or consequential damages arising out of the use
of this library and its documentation.
*************************************************************************************/
#include <stdio.h>

extern "C" { 
#include "jmemdstsrc.h"
} 

#include <string>
using std::string;

namespace Jpeg
{
	bool  ReadFileParams(const string& fileName, int& width, int& height, int& nChannels)
	{
		if (fileName.empty())
			return false;

		FILE* infile;
		if ((infile = fopen(fileName.c_str(), "rb")) == 0) 
			return false;
			
		struct jpeg_decompress_struct cinfo;
		struct jpeg_error_mgr jerr;
		cinfo.err = jpeg_std_error(&jerr);
		jpeg_create_decompress(&cinfo);

		jpeg_stdio_src(&cinfo, infile);
		jpeg_read_header(&cinfo, true);
		jpeg_calc_output_dimensions(&cinfo);
		
		width = cinfo.image_width;
		height = cinfo.image_height;
		nChannels = cinfo.num_components;
		jpeg_destroy_decompress(&cinfo);
		fclose(infile);
		return true;
	}

	bool DecompressFile(const string& fileName, unsigned char* dstBuffer, int& width, int& height, int& nChannels)
	{
		if (!dstBuffer || fileName.empty())
			return false;

		FILE* infile;
		if ((infile = fopen(fileName.c_str(), "rb")) == 0) 
			return false;
		
		struct jpeg_error_mgr jerr;
		struct jpeg_decompress_struct cinfo;
		cinfo.err = jpeg_std_error(&jerr);
		jpeg_create_decompress(&cinfo);
		jpeg_stdio_src(&cinfo, infile);
		jpeg_read_header(&cinfo, true);
		jpeg_start_decompress(&cinfo);

		width = cinfo.output_width; 
		height = cinfo.output_height;
		nChannels = cinfo.num_components;	

		unsigned char* rowptr[1];
		while (cinfo.output_scanline < cinfo.output_height)
		{
			rowptr[0] = &dstBuffer[cinfo.output_scanline*cinfo.output_width*cinfo.num_components];
			jpeg_read_scanlines(&cinfo, rowptr, 1);
		}
		jpeg_finish_decompress(&cinfo);
		jpeg_destroy_decompress(&cinfo);
		fclose(infile);
		return true;
	}	

	bool DecompressBuffer(unsigned char* srcBuffer, int size, unsigned char* dstBuffer, int& width, int& height, int& bpp)
	{
		if (!dstBuffer)
			return false;

		struct jpeg_error_mgr jerr;
		struct jpeg_decompress_struct cinfo;

		cinfo.err = jpeg_std_error(&jerr);
		jpeg_create_decompress(&cinfo);

		j_mem_src (&cinfo, srcBuffer, size);

		jpeg_read_header(&cinfo,true);
		jpeg_start_decompress(&cinfo);
		
		width = cinfo.output_width;
		height = cinfo.output_height;
		bpp = cinfo.num_components;
		unsigned char* rowptr[1];
		while (cinfo.output_scanline < cinfo.output_height)
		{
			rowptr[0] = &dstBuffer[cinfo.output_scanline*cinfo.output_width*cinfo.num_components];
			if (rowptr[0] == 0)
			{
				jpeg_finish_decompress(&cinfo);
				jpeg_destroy_decompress(&cinfo);
				return false;
			}
			if (jpeg_read_scanlines(&cinfo, rowptr, 1) != 1)
			{
				jpeg_finish_decompress(&cinfo);
				jpeg_destroy_decompress(&cinfo);
				return false;
			}
		}
		jpeg_finish_decompress(&cinfo);
		jpeg_destroy_decompress(&cinfo);
		return true;
	}

	bool CompressToFile(unsigned char* buffer, int width, int height, int bpp, const string& fileName, int quality=75)
	{
		// check if input parameters are valid
		if (fileName.empty() || !buffer || (bpp != 1 && bpp != 3) )
			return false;
		
		// create the destination file
		FILE* outfile = fopen(fileName.c_str(), "wb");
		if (outfile == 0) 
			return false;

		// create access to source buffer as expected by jpeglib
		JSAMPROW row_pointer[1];	
		int row_stride = width*bpp;	

		// create compress structure
		struct jpeg_compress_struct cinfo;
		struct jpeg_error_mgr jerr;
		jpeg_create_compress(&cinfo);
		cinfo.err = jpeg_std_error(&jerr);

		// set the known parameters and default parameters
		if (bpp == 3)
			cinfo.in_color_space = JCS_RGB; 
		else if (bpp == 1)
			cinfo.in_color_space = JCS_GRAYSCALE; 
		cinfo.image_width = width;
		cinfo.image_height = height;
		cinfo.input_components = bpp;
		jpeg_set_defaults(&cinfo);
	    jpeg_set_quality(&cinfo, quality, true);
		jpeg_stdio_dest(&cinfo, outfile);

		// decompress the data line by line
		jpeg_start_compress(&cinfo, true);
		while (cinfo.next_scanline < cinfo.image_height) 
		{
			row_pointer[0] = &buffer[cinfo.next_scanline * row_stride];
			jpeg_write_scanlines(&cinfo, row_pointer, 1);
		}

		// release structures
		jpeg_finish_compress(&cinfo);
		jpeg_destroy_compress(&cinfo);

		fclose(outfile);
		return true;
	}

	bool CompressToBuffer(unsigned char* srcBuffer, int width, int height, int bpp, unsigned char* dstBuffer, int& len, int quality)
	{
		if (!srcBuffer || !dstBuffer || len<=0 )
			return false;

		JSAMPROW row_pointer[1];	
		int row_stride = width*bpp;	

		struct jpeg_error_mgr jerr;
		struct jpeg_compress_struct cinfo;
		jpeg_create_compress(&cinfo);
		cinfo.err = jpeg_std_error(&jerr);
		if (bpp == 3)
			cinfo.in_color_space = JCS_RGB; 
		else if (bpp == 1)
			cinfo.in_color_space = JCS_GRAYSCALE; 
		cinfo.image_width = width;
		cinfo.image_height = height;
		cinfo.input_components = bpp;
		jpeg_set_defaults(&cinfo);
	    jpeg_set_quality(&cinfo, quality, true);

		j_mem_dest(&cinfo,reinterpret_cast<void**>(&dstBuffer),reinterpret_cast<unsigned int*>(&len));
		jpeg_start_compress(&cinfo,true);
		while (cinfo.next_scanline < cinfo.image_height) 
		{
			row_pointer[0] = &srcBuffer[cinfo.next_scanline * row_stride];
			jpeg_write_scanlines(&cinfo, row_pointer, 1);
		}
		jpeg_finish_compress(&cinfo);
		jpeg_destroy_compress(&cinfo);
		return true;
	}
}
