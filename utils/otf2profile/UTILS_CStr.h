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

#ifndef UTILS_CSTR_H
#define UTILS_CSTR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @file            UTILS_CStr.h
 *
 * @brief           Declare helper functions for string handling
 */

/**
 * Defines a C value analog to std::string::npos
 */
#define UTILS_CStr_npos PACKAGE_MANGLE_NAME( UTILS_CStr_npos )
extern const size_t UTILS_CStr_npos;

/**
 * Duplicates the string @a source.
 */
#define UTILS_CStr_dup PACKAGE_MANGLE_NAME( UTILS_CStr_dup )
char*
UTILS_CStr_dup( const char* source );

/**
   Searches for the first appearance of substring @a pattern in the string
   @a str after position @a pos.
   @param str     A string where the function tries to find the given pattern.
   @param pattern The string which is searched for in @a str.
   @param pos     The index of the first character in @a str, where the search
                  is started.
   @return If an appearance of @a pattern is found in @a str. It returns the
           index of the first letter of the first appearance of @a pattern in
           @a str. If the pattern is not found in @a str, it returns
           UTILS_CStr_npos.
 */
#define UTILS_CStr_find PACKAGE_MANGLE_NAME( UTILS_CStr_find )
size_t
UTILS_CStr_find( const char* str,
                 const char* pattern,
                 size_t      pos );

#ifdef __cplusplus
}
#endif

#endif /* UTILS_CSTR_H */
