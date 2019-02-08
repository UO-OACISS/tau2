/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2014
 * Forschungszentrum Juelich GmbH, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
 *
 */
/****************************************************************************
**  SCALASCA    http://www.scalasca.org/                                   **
**  KOJAK       http://www.fz-juelich.de/jsc/kojak/                        **
*****************************************************************************
**  Copyright (c) 1998-2009                                                **
**  Forschungszentrum Juelich, Juelich Supercomputing Centre               **
**                                                                         **
**  See the file COPYRIGHT in the package base directory for details       **
****************************************************************************/
#ifndef OPARI2_CTC_PARSER_H
#define OPARI2_CTC_PARSER_H

/**
 * @file    opari2_ctc_parser.h
 * @date    Started Tue Mar 25 2014
 *
 * @brief
 *
 */

#include <stddef.h>
#include <stdbool.h>

#include "opari2_region_info.h"

/** @brief CTCData */
typedef struct
{
    /** pointer to source info */
    char*               mCTCStringToParse;
    /** memory string*/
    char*               mCTCStringMemory;
    /** error string*/
    char*               mCTCStringForErrorMsg;
    /** structured region information */
    OPARI2_Region_info* mRegionInfo;
} CTCData;

/** @brief errors the user is responsible for, i.e. just errors in
 *   the passed string */
typedef enum /* CTC_ERROR_Type */
{
    CTC_ERROR_Ended_unexpectedly,
    CTC_ERROR_No_region_type,
    CTC_ERROR_No_separator_after_length_field,
    CTC_ERROR_Num_sections_invalid,
    CTC_ERROR_SCL_broken,
    CTC_ERROR_SCL_line_number_error,
    CTC_ERROR_Unknown_token,
    CTC_ERROR_Unsigned_expected,
    CTC_ERROR_User_region_name_missing,
    CTC_ERROR_Wrong_clause_value,
    CTC_ERROR_Unknown_region_type,
    CTC_ERROR_No_key,
    CTC_ERROR_No_value,
    CTC_ERROR_Zero_length_key,
    CTC_ERROR_Zero_length_value,
    CTC_ERROR_Unknown_schedule_type,
    CTC_ERROR_Unknown_default_sharing_type,
    CTC_ERROR_SCL_error,
    CTC_ERROR_Inconsistent_line_numbers
} CTC_ERROR_Type;

/** @brief print error information*/
void
OPARI2_CTC_error( CTCData*       obj,
                  CTC_ERROR_Type errorType,
                  const char*    info1 );

void
OPARI2_CTC_initCTCData( CTCData*   obj,
                        const char string[] );

void
OPARI2_CTC_parseCTCStringAndAssignRegionInfoValues( CTCData * obj,
                                                    bool ( * checkToken )( int,
                                                                           char*,
                                                                           CTCData* ) );

void
OPARI2_CTC_assignUnsigned( CTCData*    obj,
                           unsigned*   anUnsigned,
                           const char* value );

void
OPARI2_CTC_assignString( char**      aString,
                         const char* value );

void
OPARI2_CTC_assignHasClause( CTCData*    obj,
                            bool*       hasClause,
                            const char* value );

void
OPARI2_CTC_checkConsistency( CTCData* obj );

void
OPARI2_CTC_freeCTCData( CTCData* obj );

void
OPARI2_CTC_freeAndReset( char** freeMe );

/* Map entry. Matches string to enum */
typedef struct
{
    /** string representation*/
    char* mString;
    /** matching region type*/
    int   mEnum;
} OPARI2_CTCMapType;

/** @brief number of entries in regionTypesMap*/
#define OPARI2_CTC_MAP_SIZE( map ) sizeof( map ) / sizeof( OPARI2_CTCMapType )

/** @brief returns the enum associated with a string */
int
OPARI2_CTC_string2Enum( const OPARI2_CTCMapType* map,
                        const size_t             n_elements,
                        const char*              string );

/** @brief returns the string associated with an enum */
const char*
OPARI2_CTC_enum2String( const OPARI2_CTCMapType* map,
                        const size_t             n_elements,
                        int                      e_in );

#endif /* OPARI2_CTC_PARSER_H */
