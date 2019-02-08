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
/**
 * @file    opari2_ctc_parser.c
 * @date    Started Tue Mar 25 2014
 *
 * @brief
 *
 */

#include <config.h>

#include "opari2_ctc_parser.h"
#include "opari2_ctc_token.h"
#include "pomp2_region_info.h"
#include "pomp2_user_region_info.h"

#include <assert.h>
#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*----------------------------------------------------------------------------*/

/** @brief print error information*/
void
OPARI2_CTC_error( CTCData*       obj,
                  CTC_ERROR_Type errorType,
                  const char*    info1 )
{
    bool abort = true;
    printf( "Error parsing ctc string:\n\"%s\"\n",
            obj->mCTCStringForErrorMsg );
    switch ( errorType )
    {
        case CTC_ERROR_Ended_unexpectedly:
            printf( "ctc string ended unexpectedly.\n" );
            break;
        case CTC_ERROR_No_region_type:
            printf( "ctc string has no region type field or value is empty.\n" );
            break;
        case CTC_ERROR_No_separator_after_length_field:
            printf( "The separator \"*\" is missing after the length field.\n" );
            break;
        case CTC_ERROR_Num_sections_invalid:
            printf( "The value of numSections must be > 0.\n" );
            break;
        case CTC_ERROR_SCL_broken:
            printf( "The required attributes sscl and/or escl contain invalid data "
                    "or are missing.\n" );
            break;
        case CTC_ERROR_SCL_line_number_error:
            printf( "sscl or escl field has invalid line number arguments (%s).\n",
                    info1 );
            break;
        case CTC_ERROR_Unknown_token:
            printf( "Token \"%s\" not known.\n", info1 );
            abort = false;
            break;
        case CTC_ERROR_Unsigned_expected:
            printf( "A value >= 0 is expected, \"%s\" is not allowed.\n", info1 );
            break;
        case CTC_ERROR_User_region_name_missing:
            printf( "The field or value \"userRegionName\" is missing.\n" );
            break;
        case CTC_ERROR_Wrong_clause_value:
            printf( "Clause field value must be \"0\" or \"1\", "
                    "\"%s\" is not allowed.\n", info1 );
            break;
        case CTC_ERROR_Unknown_region_type:
            printf( "Region type \"%s\" not known.\n", info1 );
            break;
        case CTC_ERROR_No_key:
            printf( "Could not detect key in \"%s\", \"=\" or \"*\" missing.\n",
                    info1 );
            break;
        case CTC_ERROR_No_value:
            printf( "Could not detect value in \"%s\", \"*\" missing.\n", info1 );
            break;
        case CTC_ERROR_Zero_length_key:
            printf( "The character sequence \"*=\" is not allowed.\n" );
            break;
        case CTC_ERROR_Zero_length_value:
            printf( "The character sequence \"=*\" is not allowed.\n" );
            break;
        case CTC_ERROR_Unknown_schedule_type:
            printf( "Schedule type \"%s\" not known.\n", info1 );
            break;
        case CTC_ERROR_Unknown_default_sharing_type:
            printf( "Argument of the default close \"%s\" of unknown type.\n", info1 );
            break;
        case CTC_ERROR_Inconsistent_line_numbers:
            printf( "Warning: line numbers not valid. Expected startLineNo1 <= startLineNo2 <= endLineNo1 <= endLineNo2 \n" );
            abort = false;
        case CTC_ERROR_SCL_error:
            printf( "Error parsing source code location, "
                    "expecting \"filename:lineNo1:lineNo2\".\n" );
            break;
        default:
            puts( "ctc internal error: unknown error type." );
    }
    if ( abort )
    {
        OPARI2_CTC_freeCTCData( obj );
        puts( "Aborting" );
        exit( 1 );
    }
}

/*----------------------------------------------------------------------------*/

static void
initRegionInfo( OPARI2_Region_info* regionInfo );

static void
copyCTCStringToInternalMemory( CTCData*    obj,
                               const char* source );

void
OPARI2_CTC_initCTCData( CTCData*   obj,
                        const char string[] )
{
    initRegionInfo( obj->mRegionInfo );

    obj->mCTCStringToParse     = 0;
    obj->mCTCStringMemory      = 0;
    obj->mCTCStringForErrorMsg = 0;
    copyCTCStringToInternalMemory( obj, string );
}


static void
initRegionInfo( OPARI2_Region_info* regionInfo )
{
    regionInfo->mStartFileName = 0;
    regionInfo->mStartLine1    = 0;
    regionInfo->mStartLine2    = 0;
    regionInfo->mEndFileName   = 0;
    regionInfo->mEndLine1      = 0;
    regionInfo->mEndLine2      = 0;
}

static void
copyCTCStringToInternalMemory( CTCData*    obj,
                               const char* source )
{
    assert( obj->mCTCStringToParse == 0 );
    assert( obj->mCTCStringMemory == 0 );
    assert( obj->mCTCStringForErrorMsg == 0 );

    const size_t nBytes = strlen( source ) * sizeof( char ) + 1;
    obj->mCTCStringMemory      = malloc( nBytes );
    obj->mCTCStringForErrorMsg = malloc( nBytes );
    strcpy( obj->mCTCStringMemory, source );
    strcpy( obj->mCTCStringForErrorMsg, source );
    obj->mCTCStringToParse = obj->mCTCStringMemory;
}

void
OPARI2_CTC_freeCTCData( CTCData* obj )
{
    OPARI2_CTC_freeAndReset( &( obj->mCTCStringMemory ) );
    OPARI2_CTC_freeAndReset( &( obj->mCTCStringForErrorMsg ) );
    obj->mCTCStringToParse = 0;
}

void
OPARI2_CTC_freeAndReset( char** freeMe )
{
    if ( *freeMe )
    {
        free( *freeMe );
        *freeMe = 0;
    }
}

static CTCToken
getCTCTokenFromString( char* token );

static void
assignSourceCodeLocation( CTCData*  obj,
                          char**    fileName,
                          unsigned* line1,
                          unsigned* line2,
                          char*     value );

static void
ignoreLengthField( CTCData* obj );

/** @brief map with CTC tokens*/
static const OPARI2_CTCMapType ctcTokenMap[] =
{
    CTC_REGION_TOKEN_MAP_ENTRIES,
    CTC_OPENMP_TOKEN_MAP_ENTRIES,
    CTC_USER_REGION_TOKEN_MAP_ENTRIES
};

static bool
getKeyValuePair( CTCData* obj,
                 char**   key,
                 char**   value );

void
OPARI2_CTC_parseCTCStringAndAssignRegionInfoValues( CTCData* obj,
                                                    bool     ( * checkToken )( int,
                                                                               char*,
                                                                               CTCData* ) )
{
    char*    key;
    char*    value;
    CTCToken token;

    ignoreLengthField( obj );

    while ( getKeyValuePair( obj, &key, &value ) )
    {
        token =
            ( CTCToken )OPARI2_CTC_string2Enum( ctcTokenMap,
                                                OPARI2_CTC_MAP_SIZE( ctcTokenMap ),
                                                key );

        switch ( token )
        {
            case CTC_Start_source_code_location:
                assignSourceCodeLocation( obj,
                                          &obj->mRegionInfo->mStartFileName,
                                          &obj->mRegionInfo->mStartLine1,
                                          &obj->mRegionInfo->mStartLine2,
                                          value );
                break;
            case CTC_End_source_code_location:
                assignSourceCodeLocation( obj,
                                          &obj->mRegionInfo->mEndFileName,
                                          &obj->mRegionInfo->mEndLine1,
                                          &obj->mRegionInfo->mEndLine2,
                                          value );
                break;
            default:
                if ( !checkToken( token, value, obj ) )
                {
                    OPARI2_CTC_error( obj, CTC_ERROR_Unknown_token, key );
                }
        }
    }
}

static void
ignoreLengthField( CTCData* obj )
{
    /* We expect ctcString to look like "42*key=value*...**"
     * The length field is redundant and we don't use it in our parsing
     * implementation. */
    while ( obj->mCTCStringToParse && isdigit( *obj->mCTCStringToParse ) )
    {
        ++( obj->mCTCStringToParse );
    }

    if ( !obj->mCTCStringToParse )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Ended_unexpectedly, 0 );
    }
    if ( *obj->mCTCStringToParse != '*' )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_No_separator_after_length_field, 0 );
    }
    ++( obj->mCTCStringToParse );
    if ( !obj->mCTCStringToParse )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Ended_unexpectedly, 0 );
    }
}


static bool
extractNextToken( char**     string,
                  const char tokenDelimiter );

static bool
getKeyValuePair( CTCData* obj,
                 char**   key,
                 char**   value )
{
    /* We expect ctcString to look like "key=value*...**" or "*".   */
    if ( *( obj->mCTCStringToParse ) == '*' )
    {
        return false; /* end of ctc string */
    }

    if ( *( obj->mCTCStringToParse ) == '\0' )
    {
        return false; /* also end of ctc string. we don't force the second "*" */
    }

    *key = obj->mCTCStringToParse;
    if ( !extractNextToken( &obj->mCTCStringToParse, '=' ) )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_No_key, *key );
    }
    if ( strlen( *key ) == 0 )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Zero_length_key, 0 );
    }

    *value = obj->mCTCStringToParse;
    if ( !extractNextToken( &obj->mCTCStringToParse, '*' ) )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_No_value, *value );
    }
    if ( strlen( *value ) == 0 )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Zero_length_value, 0 );
    }
    /*@ in the hasSchedule clause was a '*' originally and needs to be replaced back*/
    if ( strcmp( *key, "hasSchedule" ) == 0 )
    {
        while ( strchr( *value, '@' ) )
        {
            ( strchr( *value, '@' ) )[ 0 ] = '*';
        }
    }
    return true;
}

static bool
extractNextToken( char**     string,
                  const char tokenDelimiter )
{
    *string = strchr( *string, tokenDelimiter );
    if ( !( *string && **string == tokenDelimiter ) )
    {
        return false;
    }
    **string = '\0'; /* extraction */
    ++( *string );
    return true;
}

static void
assignSourceCodeLocation( CTCData*  obj,
                          char**    filename,
                          unsigned* line1,
                          unsigned* line2,
                          char*     value )
{
    /* We assume that value looks like "foo.c:42:43" */
    char* token    = value;
    int   line1Tmp = -1;
    int   line2Tmp = -1;
    bool  continueExtraction;
    assert( *filename == 0 );

    if ( ( continueExtraction = extractNextToken( &value, ':' ) ) )
    {
        *filename = malloc( strlen( token ) * sizeof( char ) + 1 );
        strcpy( *filename, token );
    }
    token = value;
    if ( continueExtraction &&
         ( continueExtraction = extractNextToken( &value, ':' ) ) )
    {
        line1Tmp = atoi( token );
    }
    token = value;
    if ( continueExtraction && extractNextToken( &value, '\0' ) )
    {
        line2Tmp = atoi( token );
    }

    if ( *filename != 0 && line1Tmp > -1 && line2Tmp > -1 )
    {
        *line1 = line1Tmp;
        *line2 = line2Tmp;
        if ( *line1 > *line2 )
        {
            OPARI2_CTC_error( obj, CTC_ERROR_SCL_line_number_error, "line1 > line2" );
        }
    }
    else
    {
        OPARI2_CTC_error( obj, CTC_ERROR_SCL_error, 0 );
    }
}

void
OPARI2_CTC_assignHasClause( CTCData*    obj,
                            bool*       hasClause,
                            const char* value )
{
    if ( !isdigit( *value ) )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Wrong_clause_value, value );
    }

    int tmp = atoi( value );
    if ( tmp != 0 && tmp != 1 )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Wrong_clause_value, value );
    }
    *hasClause = tmp;
}

void
OPARI2_CTC_assignUnsigned( CTCData*    obj,
                           unsigned*   anUnsigned,
                           const char* value )
{
    int tmp = atoi( value );
    if ( tmp < 0 )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Unsigned_expected, value );
    }
    *anUnsigned = tmp;
}

void
OPARI2_CTC_assignString( char**      aString,
                         const char* value )
{
    *aString = malloc( strlen( value ) * sizeof( char ) + 1 );
    strcpy( *aString, value );
}

void
OPARI2_CTC_checkConsistency( CTCData* obj )
{
    bool requiredAttributesFound;

    requiredAttributesFound = ( obj->mRegionInfo->mStartFileName &&
                                obj->mRegionInfo->mEndFileName );
    if ( !requiredAttributesFound )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_SCL_broken, 0 );
        return;
    }

    if ( obj->mRegionInfo->mStartLine1 > obj->mRegionInfo->mStartLine2 )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Inconsistent_line_numbers, 0 );
        return;
    }

    if ( obj->mRegionInfo->mEndLine1 > obj->mRegionInfo->mEndLine2 )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Inconsistent_line_numbers, 0 );
        return;
    }
}

/*----------------------------------------------------------------------------*/
int
OPARI2_CTC_string2Enum( const OPARI2_CTCMapType* map,
                        const size_t             n_elements,
                        const char*              string )
{
    int i;
    for ( i = 0; i < n_elements; ++i )
    {
        if ( strcmp( map[ i ].mString, string ) == 0 )
        {
            return map[ i ].mEnum;
        }
    }

    return 0;
}

/** @brief returns the string associated with an enum */
const char*
OPARI2_CTC_enum2String( const OPARI2_CTCMapType* map,
                        const size_t             n_elements,
                        int                      e_in )
{
    int i;
    for ( i = 0; i < n_elements; ++i )
    {
        if ( e_in == map[ i ].mEnum )
        {
            return map[ i ].mString;
        }
    }
    return "no valid region type";
}
