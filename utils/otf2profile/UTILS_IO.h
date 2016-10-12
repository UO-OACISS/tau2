/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2012,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2012,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2012, 2014,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2012,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2012,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2012,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2012,
 * Technische Universitaet Muenchen, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license.  See the COPYING file in the package base
 * directory for details.
 *
 */

#ifndef UTILS_IO_H
#define UTILS_IO_H

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef __cplusplus
#include <stdbool.h>
#endif
#include <stddef.h>
#include <stdio.h>

#include <UTILS_Error.h>

/**
   Checks whether a file exists.
   @param filename  The name of the file which is checked.
   @returns true if the file exists, else false.
 */
#define UTILS_DoesFileExist PACKAGE_MANGLE_NAME( UTILS_DoesFileExist )
bool
UTILS_DoesFileExist( const char* filename );

/**
   Gives the path to the executeable name (argv[0]) without the executable
   name and trailing slash. Checks whether the given name contains path
   information. If not, it searches the directories in the PATH environment
   variable.
   @param exe    The executable name as given to argv[0].
   @returns The path to the executable without the executable name itselfs and
            trailing slash. If no path was found, NULL is retured.
 */
#define UTILS_GetExecutablePath PACKAGE_MANGLE_NAME( UTILS_GetExecutablePath )
char*
UTILS_GetExecutablePath( const char* exe );

/**
 * Reads a line until a newline from @a file. The line is stored in the buffer
 * pointed to by @a buffer. If the buffer is not large enough, the buffer is reallocated
 * to contain the whole line. The current size of the buffer is stored in @a buffer_size.
 * @param buffer      Pointer to a storage location, where the pointer to an allocated
 *                    memory block is stored, to which the read line written. If the
 *                    memory block is too small it will be enlarged. The buffer must be
 *                    freed by the calling function. The pointer to the buffer must not
 *                    be NULL. However, *buffer may be NULL.
 * @param buffer_size Points to a memory location where the size of the memory block
 *                    pointed to by @a **buffer is stored. If the memory is resized,
 *                    the buffer_size is updated.
 * @param file        A file handle of an already open file, from which a line is read.
 * @returns PACKAGE_SUCCESS, if a line was read successfully. If the end of the file
 *          was reached, PACKAGE_ERROR_END_OF_BUFFER is returned. If an error occured,
 *          an error code is returned. Possible error codes are:
 *          PACKAGE_ERROR_MEM_ALLOC_FAILED and PACKAGE_ERROR_FILE_INTERACTION.
 */
#define UTILS_IO_GetLine PACKAGE_MANGLE_NAME( UTILS_IO_GetLine )
PACKAGE_ErrorCode
UTILS_IO_GetLine( char**  buffer,
                  size_t* buffer_size,
                  FILE*   file );

/**
 * Returns true if @a path is a relative or absolute path.
 */
#define UTILS_IO_HasPath PACKAGE_MANGLE_NAME( UTILS_IO_HasPath )
bool
UTILS_IO_HasPath( const char* path );

/**
 * Cuts of the path prefix from a filename.
 * @param path a filename which might have a path prefix.
 * @returns a pointer to the position in @a path where the filename starts.
 */
#define UTILS_IO_GetWithoutPath PACKAGE_MANGLE_NAME( UTILS_IO_GetWithoutPath )
const char*
UTILS_IO_GetWithoutPath( const char* path );

/**
 * Simplifies the @a path. If @path contains ../ subpathes they are merged with
 * preceding pathes. Thus the final representation of the path is minimized. Furthermore,
 * /./ sequences are cut out.
 * If the original path contained at least one slash, this functions will prepend
 * a ./ if the simplified path has no nore slashes. This maintains the information
 * whether the sting was a simple file name or given as a path.
 * @param path the path that is simplified. The simplified path is stored in the same
 *        location. It must not be a pointer to a constant in the program segment.
 *        Passing a pointer to a string literal results in a segmentation fault.
 */
#define UTILS_IO_SimplifyPath PACKAGE_MANGLE_NAME( UTILS_IO_SimplifyPath )
void
UTILS_IO_SimplifyPath( char* path );

/**
 * Joins an arbitrary number of path elements into one path.
 *
 * If a path element is an absolute path, any previous path elements are
 * discarded. If an path element is empty (ie. "") it is ignored.
 *
 * @param nPaths The number of variable argument entries following the
 *               @a nPaths argument.
 * @param ...    Path elements to be joined. All of type <code>const char*</code>.
 *
 * @return Joined path, allocated with @a malloc.
 */
#define UTILS_IO_JoinPath PACKAGE_MANGLE_NAME( UTILS_IO_JoinPath )
char*
UTILS_IO_JoinPath( int nPaths,
                   ... );

/**
 * Wrapper function for gethostname which is not declared or implemented on all systems
 * @param name     Pointer to a memory location of length @a namelen where the name
 *                 is stored. The result is NULL-terminated, except when the name was
 *                 too long for the memory location.
 * @param namelen  Length of the memory segment reserved for @a name.
 * @returns zero if the operation was successful. Else a non-zero value is returned.
 */
#define UTILS_IO_GetHostname PACKAGE_MANGLE_NAME( UTILS_IO_GetHostname )
int
UTILS_IO_GetHostname( char*  name,
                      size_t namelen );

/**
 * Wrapper function for getcwd which is not declared or implemented on all systems.
 * Uses the same syntax as the POSIX getcwd.
 * @param buf  Points to the buffer to copy the current working directory to,
 *             of NULL if getcwd() should allocate the buffer.
 * @param size Is the size, in bytes, of the array of characters that buf points to.
 */
#define UTILS_IO_GetCwd PACKAGE_MANGLE_NAME( UTILS_IO_GetCwd )
char*
UTILS_IO_GetCwd( char*  buf,
                 size_t size );

#ifdef __cplusplus
}
#endif


#endif /* UTILS_IO_H */
