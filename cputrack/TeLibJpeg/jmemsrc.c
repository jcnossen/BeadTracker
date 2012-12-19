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
#include <stdio.h>
#include <stdlib.h>
#include "jmemdstsrc.h"
/*
 Initialize source --- Nothing to do
 */
METHODDEF(void)
init_source (j_decompress_ptr cinfo)
{}


/*
 Fill the input buffer --- called whenever buffer is emptied.
 */
METHODDEF(boolean)
fill_input_buffer (j_decompress_ptr cinfo)
{
	my_src_ptr src = (my_src_ptr) cinfo->src;

	src->pub.next_input_byte = src->buffer;
	src->pub.bytes_in_buffer = src->bufsize;

	return TRUE;
}

/*
 Skip data --- used to skip over a potentially large amount of
 uninteresting data.
 */
METHODDEF(void)
skip_input_data (j_decompress_ptr cinfo, long num_bytes)
{
	my_src_ptr src = (my_src_ptr) cinfo->src;

	/* just move the ptr */
	src->pub.next_input_byte += num_bytes;
	src->pub.bytes_in_buffer -= num_bytes;
}

/*
  Terminate source --- called by jpeg_finish_decompress
 */
METHODDEF(void)
term_source (j_decompress_ptr cinfo)
{}

/*
 Prepare for input from a memory buffer.
 */
GLOBAL(void)
j_mem_src (j_decompress_ptr cinfo, unsigned char* buffer, unsigned int bufsize)
{
	my_src_ptr src;

	if (cinfo->src == NULL) {
		cinfo->src = (struct jpeg_source_mgr *)
		(*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT,  sizeof(my_source_mgr));
  }

	src = (my_src_ptr) cinfo->src;
	src->pub.init_source = init_source;
	src->pub.fill_input_buffer = fill_input_buffer;
	src->pub.skip_input_data = skip_input_data;
	src->pub.resync_to_restart = jpeg_resync_to_restart; 
	src->pub.term_source = term_source;
	src->pub.bytes_in_buffer = 0; 
	src->pub.next_input_byte = NULL; 

	src->buffer = buffer;
	src->bufsize = bufsize;
}

