/************************************************************************************
TerraLib - a library for developing GIS applications.
Copyright © 2001-2007 INPE and Tecgraf/PUC-Rio.

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

/* this is not a core library module, so it doesn't define JPEG_INTERNALS */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "jmemdstsrc.h"
/*
 Initialize destination 
 */
METHODDEF(void)
init_destination (j_compress_ptr cinfo)
{
	mem_dest_ptr dest = (mem_dest_ptr) cinfo->dest;

	/* have the compresser write the image target memory */
	dest->pub.next_output_byte = (*(dest->pTargetData));
	dest->pub.free_in_buffer = dest->initialDataSize;
}


/* 
 * change_target_buffer_size() enlarges pTargetData if needed
 * it is used by empty_output_buffer() and term_destination()
 */
void change_target_buffer_size(j_compress_ptr cinfo, unsigned int new_size)
{
	mem_dest_ptr dest = (mem_dest_ptr)cinfo->dest;
	JOCTET * new_ptr = NULL;
	JOCTET * old_ptr = *(dest->pTargetData);
	if(new_size == dest->initialDataSize)
      return;
	if(new_size < dest->initialDataSize && dest->bufferPreallocated && !dest->bufferSizeChanged)
      return;
	new_ptr = (JOCTET*)malloc(new_size);
	if (new_ptr)
	{
		dest->initialDataSize = new_size;
		dest->bufferSizeChanged = 1;
		memcpy(new_ptr, old_ptr, *(dest->pNumBytes));
		free(old_ptr);
		*(dest->pTargetData) = new_ptr;
   }
   else if(!dest->bufferPreallocated)
   {
		free(old_ptr);
		dest->initialDataSize = 0;
		*(dest->pTargetData) = NULL;
		*(dest->pNumBytes) = 0;
//      ERROR!
   }
}

/*
 * Empty the output buffer --- called whenever buffer fills up.
 */
METHODDEF(boolean)
empty_output_buffer (j_compress_ptr cinfo)
{
	mem_dest_ptr	dest = (mem_dest_ptr) cinfo->dest;
	*(dest->pNumBytes) = dest->initialDataSize;

	change_target_buffer_size(cinfo, dest->initialDataSize + OUTPUT_BUF_SIZE);

	dest->pub.next_output_byte = *(dest->pTargetData) + *(dest->pNumBytes);
	dest->pub.free_in_buffer = OUTPUT_BUF_SIZE;

	return TRUE;
}

/*
 * Terminate destination --- called by jpeg_finish_compress
 */
METHODDEF(void)
term_destination (j_compress_ptr cinfo)
{
	mem_dest_ptr dest = (mem_dest_ptr)cinfo->dest;
	if (dest->bufferSizeChanged)
	{
		unsigned int datacount = OUTPUT_BUF_SIZE - dest->pub.free_in_buffer;
		*(dest->pNumBytes) += datacount;
		change_target_buffer_size(cinfo, *(dest->pNumBytes));
	}
	else
	{
	   *(dest->pNumBytes) = dest->initialDataSize - dest->pub.free_in_buffer;
	}
   return;
}

/*
 * Prepare for output to an allocated buffer.
 * The caller is responsible for free()ing the buffer when they are done.
 */

GLOBAL(void)
j_mem_dest(j_compress_ptr cinfo, void **pTargetData, unsigned int *pNumBytes)
{
	mem_dest_ptr dest;	if(cinfo->dest == NULL)
	{
		cinfo->dest = (struct jpeg_destination_mgr *)
        (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT,
        sizeof(mem_destination_mgr));
	}

	dest = (mem_dest_ptr) cinfo->dest;
	dest->pub.init_destination = init_destination;
	dest->pub.empty_output_buffer = empty_output_buffer;
	dest->pub.term_destination = term_destination;
   /* if the number of bytes > 0 and the data pointer is not NULL, then we
      assume that space has been allocated, otherwise, we allocate here */
	if (( *pNumBytes > 0 ) && (*pTargetData ))
	{
		dest->bufferPreallocated = 1;
		dest->initialDataSize = *pNumBytes;
	}
	else
	{
		dest->bufferPreallocated = 0;
		if(!((*pTargetData) = malloc(OUTPUT_BUF_SIZE * sizeof(JOCTET))))
		{
         /* malloc() failed - call jpeg error/exit subsystem */
		// ERROR( cinfo, JERR_OUT_OF_MEMORY, 0);
		}
		else
		{
			dest->initialDataSize = OUTPUT_BUF_SIZE;
		}
   }

   dest->pNumBytes = pNumBytes;
   dest->pTargetData = (JOCTET**)pTargetData;
   dest->bufferSizeChanged = 0;
   *pNumBytes = 0;
	return;
}
