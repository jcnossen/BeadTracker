/************************************************************************************
TerraLib - a library for developing GIS applications.
Copyright � 2001-2004 INPE and Tecgraf/PUC-Rio.

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
/*! \file TeLibJpegWrapper.h
    This file is a wrapper around libjpeg provinding higher level C++ calls to
	some functionalities.
*/
#ifndef  __TERRALIB_DRIVER_LIBJPEG_WRAPPER_H
#define  __TERRALIB_DRIVER_LIBJPEG_WRAPPER_H

#include "../../kernel/TeDefines.h"
#include <string>
/** 
@brief A wrapper around the libjpeg provinding higher level C++ functions
*/
namespace Jpeg
{
	/** Reads the main informations about a JPEG image file 
	  \param fileName	name of the file 
      \param width		return the number of columns of the data
      \param height		return the number of lines of the data
	  \param nChannels	return the number of bands, or channels, of the data
	  \return true or false whether the raster was imported successfully
	*/
	TL_DLL bool ReadFileParams(const std::string& fileName, int& width, int& height, int& nChannels);

// --------------  Handles the compression and uncompression to/from files.

	/** Reads and decompresses a JPEG image file to a buffer in memory 
	  \param fileName	name of the file 
	  \param dstBuffer	pointer to a buffer to return the decompressed data. The function 
						assumes that it was allocated with enough space to hold the decompressed data
      \param width		returns the number of columns of the data
      \param height		returns the number of lines of the data
	  \param nChannels	returns the number of bands, or channels, of the data
	  \return true or false whether the decompressing was successfull or not
	*/
	TL_DLL bool DecompressFile(const std::string& fileName, unsigned char* dstBuffer, int& width, int& height, int& nChannels);

	/** Compresses an image buffer to a JPEG image file 
      \param buffer		address of the buffer that contains the image in memory
	  \param width		width of image in pixels
      \param height		height of image in pixels
	  \param bpp		number of bytes per pixel (1 or 3)
	  \param fileName	name of the compressed file 
	  \param quality	image quality as a percentage value
	  \return true or false whether the compressing was successfull or not
	*/
	TL_DLL bool CompressToFile(unsigned char* buffer, int width, int height, int bpp, const std::string& fileName, int quality=75);

	/** Decompresses a JPEG image buffer to a buffer in memory 
	  \param srcBuffer	memory address containing jpeg compressed data 
	  \param size		size in bytes of the jpeg compressed data
	  \param dstBuffer	pointer to a buffer to return the decompressed data. The function 
						assumes that it was allocated with enough space to hold the decompressed data
      \param width		return the number of columns of the data
      \param height		return the number of lines of the data
	  \param bpp		return the number of bytes per pixel
	  \return true or false whether the decompressing was successfull or not
	*/
	TL_DLL bool DecompressBuffer(unsigned char* srcBuffer, int size, unsigned char* dstBuffer, int& width, int& height, int& bpp); 

	/** Compresses an image buffer to a JPEG image in memory 
      \param srcBuffer	address of the image in memory
	  \param width		width of image in pixels
      \param height		height of image in pixels
	  \param bpp		number of bytes per pixel (1 or 3)
	  \param dstBuffer	pointer to a buffer to buffer to return the compressed data. The function assumes that it was allocated 
						with enough space to hold the compressed data
	  \param len		returns the size of the compressed data. Initially this parameter should contain 
						the size of the pre-allocated buffer
	  \param quality	image quality as a percentage
	  \return true or false whether the compressing was successfull or not
	*/
	TL_DLL bool CompressToBuffer(unsigned char* srcBuffer, int width, int height, int bpp, unsigned char* dstBuffer, int& len, int quality=75); 
}
#endif

