/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2011,
 *    RWTH Aachen University, Germany
 *    Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *    Technische Universitaet Dresden, Germany
 *    University of Oregon, Eugene, USA
 *    Forschungszentrum Juelich GmbH, Germany
 *    German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *    Technische Universitaet Muenchen, Germany
 *
 * See the COPYING file in the package base directory for details.
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
 * @file    pomp2_region_info.c
 * @author  Christian R&ouml;ssel <c.roessel@fz-juelich.de>
 * @date    Started Fri Mar 20 16:59:41 2009
 *
 * @brief
 *
 */

#include <config.h>

#include "pomp2_region_info.h"

#include <assert.h>
#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*----------------------------------------------------------------------------*/

/** @brief CTCData */
typedef struct
{
    /** structured region information */
    POMP2_Region_info* mRegionInfo;
    /** CTC String representation*/
    char*              mCTCStringToParse;
    /** memory string*/
    char*              mCTCStringMemory;
    /** error string*/
    char*              mCTCStringForErrorMsg;
} CTCData;

/** @brief errors the user is responsibel for, i.e. just errors in
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
    CTC_ERROR_SCL_error,
    CTC_ERROR_Inconsistent_line_numbers
} CTC_ERROR_Type;

static void
freeCTCData( CTCData* obj );

/** @brief print error information*/
void
ctcError( CTCData*       obj,
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
        case CTC_ERROR_Inconsistent_line_numbers:
            printf( "Line numbers not valid. Expected startLineNo1 <= startLineNo2 <= endLineNo1 <= endLineNo2 \n" );
        case CTC_ERROR_SCL_error:
            printf( "Error parsing source code location, "
                    "expecting \"filename:lineNo1:lineNo2\".\n" );
            break;
        default:
            puts( "ctc internal error: unknown error type." );
    }
    if ( abort )
    {
        freeCTCData( obj );
        puts( "Aborting" );
        exit( 1 );
    }
}

/*----------------------------------------------------------------------------*/

static void
parseCTCStringAndAssignRegionInfoValues( CTCData* obj );
static void
checkConsistency( CTCData* obj );
static void
initCTCData( CTCData*           obj,
             const char         string[],
             POMP2_Region_info* regionInfo );


void
ctcString2RegionInfo( const char         string[],
                      POMP2_Region_info* regionInfo )
{
    assert( regionInfo );

    CTCData ctcData;
    initCTCData( &ctcData, string, regionInfo );
    parseCTCStringAndAssignRegionInfoValues( &ctcData );
    checkConsistency( &ctcData );
    freeCTCData( &ctcData );
}


static void
initRegionInfo( CTCData* obj );
static void
copyCTCStringToInternalMemory( CTCData*    obj,
                               const char* source );

static void
initCTCData( CTCData*           obj,
             const char         string[],
             POMP2_Region_info* regionInfo )
{
    obj->mRegionInfo = regionInfo;
    initRegionInfo( obj );

    obj->mCTCStringToParse     = 0;
    obj->mCTCStringMemory      = 0;
    obj->mCTCStringForErrorMsg = 0;
    copyCTCStringToInternalMemory( obj, string );
}


static void
initRegionInfo( CTCData* obj )
{
    obj->mRegionInfo->mRegionType      = POMP2_No_type;
    obj->mRegionInfo->mStartFileName   = 0;
    obj->mRegionInfo->mStartLine1      = 0;
    obj->mRegionInfo->mStartLine2      = 0;
    obj->mRegionInfo->mEndFileName     = 0;
    obj->mRegionInfo->mEndLine1        = 0;
    obj->mRegionInfo->mEndLine2        = 0;
    obj->mRegionInfo->mHasCollapse     = false;
    obj->mRegionInfo->mHasCopyIn       = false;
    obj->mRegionInfo->mHasCopyPrivate  = false;
    obj->mRegionInfo->mHasFirstPrivate = false;
    obj->mRegionInfo->mHasIf           = false;
    obj->mRegionInfo->mHasLastPrivate  = false;
    obj->mRegionInfo->mHasNoWait       = false;
    obj->mRegionInfo->mHasNumThreads   = false;
    obj->mRegionInfo->mHasOrdered      = false;
    obj->mRegionInfo->mHasReduction    = false;
    obj->mRegionInfo->mHasUntied       = false;
    obj->mRegionInfo->mScheduleType    = POMP2_No_schedule;
    obj->mRegionInfo->mNumSections     = 0;
    obj->mRegionInfo->mCriticalName    = 0;
    obj->mRegionInfo->mUserRegionName  = 0;
    obj->mRegionInfo->mUserGroupName   = 0;
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

static void
freeAndReset( char** freeMe );

static void
freeCTCData( CTCData* obj )
{
    freeAndReset( &( obj->mCTCStringMemory ) );
    freeAndReset( &( obj->mCTCStringForErrorMsg ) );
    obj->mCTCStringToParse = 0;
}

static void
freeAndReset( char** freeMe )
{
    if ( *freeMe )
    {
        free( *freeMe );
        *freeMe = 0;
    }
}

/** CTC Tokens */
typedef enum
{
    CTC_Region_type,
    CTC_Start_source_code_location,
    CTC_End_source_code_location,
    CTC_Has_copy_in,
    CTC_Has_copy_private,
    CTC_Has_first_private,
    CTC_Has_last_private,
    CTC_Has_no_wait,
    CTC_Has_ordered,
    CTC_Has_reduction,
    CTC_Schedule_type,
    CTC_Num_sections,
    CTC_Critical_name,
    CTC_User_region_name,
    CTC_User_group_name,
    CTC_Has_if,
    CTC_Has_collapse,
    CTC_Has_num_threads,
    CTC_Has_untied,
    CTC_No_token
} CTCToken;

static void
ignoreLengthField( CTCData* obj );
static bool
getKeyValuePair( CTCData* obj,
                 char**   key,
                 char**   value );
static CTCToken
getCTCTokenFromString( char* token );
static void
assignRegionType( CTCData*    obj,
                  const char* value );
static void
assignSourceCodeLocation( CTCData*  obj,
                          char**    fileName,
                          unsigned* line1,
                          unsigned* line2,
                          char*     value );
static void
assignHasClause( CTCData*    obj,
                 bool*       hasClause,
                 const char* value );
static void
assignScheduleType( CTCData*       obj,
                    char* restrict value );
static void
assignUnsigned( CTCData*    obj,
                unsigned*   anUnsigned,
                const char* value );
static void
assignString( char**      aString,
              const char* value );

static void
parseCTCStringAndAssignRegionInfoValues( CTCData* obj )
{
    char* key;
    char* value;

    ignoreLengthField( obj );

    while ( getKeyValuePair( obj, &key, &value ) )
    {
        switch ( getCTCTokenFromString( key ) )
        {
            case CTC_Region_type:
                assignRegionType( obj, value );
                break;
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
            case CTC_Has_copy_in:
                assignHasClause( obj, &obj->mRegionInfo->mHasCopyIn, value );
                break;
            case CTC_Has_copy_private:
                assignHasClause( obj, &obj->mRegionInfo->mHasCopyPrivate, value );
                break;
            case CTC_Has_first_private:
                assignHasClause( obj, &obj->mRegionInfo->mHasFirstPrivate, value );
                break;
            case CTC_Has_if:
                assignHasClause( obj, &obj->mRegionInfo->mHasIf, value );
                break;
            case CTC_Has_last_private:
                assignHasClause( obj, &obj->mRegionInfo->mHasLastPrivate, value );
                break;
            case CTC_Has_no_wait:
                assignHasClause( obj, &obj->mRegionInfo->mHasNoWait, value );
                break;
            case CTC_Has_num_threads:
                assignHasClause( obj, &obj->mRegionInfo->mHasNumThreads, value );
                break;
            case CTC_Has_ordered:
                assignHasClause( obj, &obj->mRegionInfo->mHasOrdered, value );
                break;
            case CTC_Has_reduction:
                assignHasClause( obj, &obj->mRegionInfo->mHasReduction, value );
                break;
            case CTC_Has_collapse:
                assignHasClause( obj, &obj->mRegionInfo->mHasCollapse, value );
                break;
            case CTC_Has_untied:
                assignHasClause(  obj, &obj->mRegionInfo->mHasUntied, value );
                break;
            case CTC_Schedule_type:
                assignScheduleType( obj, value );
                break;
            case CTC_Num_sections:
                assignUnsigned( obj, &obj->mRegionInfo->mNumSections, value );
                break;
            case CTC_Critical_name:
                assignString( &obj->mRegionInfo->mCriticalName, value );
                break;
            case CTC_User_region_name:
                assignString( &obj->mRegionInfo->mUserRegionName, value );
                break;
            case CTC_User_group_name:
                assignString( &obj->mRegionInfo->mUserGroupName, value );
                break;
            default:
                ctcError( obj, CTC_ERROR_Unknown_token, key );
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
        ctcError( obj, CTC_ERROR_Ended_unexpectedly, 0 );
    }
    if ( *obj->mCTCStringToParse != '*' )
    {
        ctcError( obj, CTC_ERROR_No_separator_after_length_field, 0 );
    }
    ++( obj->mCTCStringToParse );
    if ( !obj->mCTCStringToParse )
    {
        ctcError( obj, CTC_ERROR_Ended_unexpectedly, 0 );
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
        ctcError( obj, CTC_ERROR_No_key, *key );
    }
    if ( strlen( *key ) == 0 )
    {
        ctcError( obj, CTC_ERROR_Zero_length_key, 0 );
    }

    *value = obj->mCTCStringToParse;
    if ( !extractNextToken( &obj->mCTCStringToParse, '*' ) )
    {
        ctcError( obj, CTC_ERROR_No_value, *value );
    }
    if ( strlen( *value ) == 0 )
    {
        ctcError( obj, CTC_ERROR_Zero_length_value, 0 );
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

/** @brief matching between string description and CTC token*/
typedef struct
{
    /** string representation*/
    char*    mTokenString;
    /** matching CTCToken*/
    CTCToken mToken;
} CTCTokenMapValueType;

/** @brief map with CTC tokens*/
static const CTCTokenMapValueType ctcTokenMap[] =
{
    /* Entries must be sorted to be used in binary search. */
    /* If you add/remove items update ctcTokenMapSize      */
    { "criticalName",    CTC_Critical_name                         },
    { "escl",            CTC_End_source_code_location              },
    { "hasCollapse",     CTC_Has_collapse                          },
    { "hasCopyIn",       CTC_Has_copy_in                           },
    { "hasCopyPrivate",  CTC_Has_copy_private                      },
    { "hasFirstPrivate", CTC_Has_first_private                     },
    { "hasIf",           CTC_Has_if                                },
    { "hasLastPrivate",  CTC_Has_last_private                      },
    { "hasNoWait",       CTC_Has_no_wait                           },
    { "hasNumThreads",   CTC_Has_num_threads                       },
    { "hasOrdered",      CTC_Has_ordered                           },
    { "hasReduction",    CTC_Has_reduction                         },
    { "hasUntied",       CTC_Has_untied                            },
    { "numSections",     CTC_Num_sections                          },
    { "regionType",      CTC_Region_type                           },
    { "scheduleType",    CTC_Schedule_type                         },
    { "sscl",            CTC_Start_source_code_location            },
    { "userGroupName",   CTC_User_group_name                       },
    { "userRegionName",  CTC_User_region_name                      }
};

/** @brief number of entries in ctcTokenMap*/
const size_t ctcTokenMapSize = sizeof( ctcTokenMap ) / sizeof( CTCTokenMapValueType );

static int
ctcTokenMapCompare( const void* searchToken,
                    const void* mapElem );

static CTCToken
getCTCTokenFromString( char* token )
{
    CTCTokenMapValueType* mapElem = ( CTCTokenMapValueType* )bsearch(
        token,
        &ctcTokenMap,
        ctcTokenMapSize,
        sizeof( CTCTokenMapValueType ),
        ctcTokenMapCompare );

    if ( mapElem )
    {
        return mapElem->mToken;
    }
    else
    {
        return CTC_No_token;
    }
}

static int
ctcTokenMapCompare( const void* searchToken,
                    const void* mapElem )
{
    const char* const     token = ( const char* )searchToken;
    CTCTokenMapValueType* elem  = ( CTCTokenMapValueType* )mapElem;

    return strcmp( token, elem->mTokenString );
}
/** @brief maching between region string and region type */
typedef struct
{
    /** string representation*/
    char*             mRegionTypeString;
    /** matching region type*/
    POMP2_Region_type mRegionType;
} RegionTypesMapValueType;

/** @brief map with region types*/
static const RegionTypesMapValueType regionTypesMap[] =
{
    /* Entries must be sorted to be used in binary search. */
    /* If you add/remove items, regionTypesMap_size  */
    { "atomic",            POMP2_Atomic                 },
    { "barrier",           POMP2_Barrier                },
    { "critical",          POMP2_Critical               },
    { "do",                POMP2_Do                     },
    { "flush",             POMP2_Flush                  },
    { "for",               POMP2_For                    },
    { "master",            POMP2_Master                 },
    { "ordered",           POMP2_Ordered                },
    { "parallel",          POMP2_Parallel               },
    { "paralleldo",        POMP2_Parallel_do            },
    { "parallelfor",       POMP2_Parallel_for           },
    { "parallelsections",  POMP2_Parallel_sections      },
    { "parallelworkshare", POMP2_Parallel_workshare     },
    { "region",            POMP2_User_region            },
    { "sections",          POMP2_Sections               },
    { "single",            POMP2_Single                 },
    { "task",              POMP2_Task                   },
    { "taskuntied",        POMP2_Taskuntied             },
    { "taskwait",          POMP2_Taskwait               },
    { "workshare",         POMP2_Workshare              }
};

/** @brief number of entries in regionTypesMap*/
const size_t regionTypesMapSize = sizeof( regionTypesMap ) / sizeof( RegionTypesMapValueType );

static POMP2_Region_type
getRegionTypeFromString( const char* regionTypeString );

static void
assignRegionType( CTCData*    obj,
                  const char* value )
{
    obj->mRegionInfo->mRegionType = getRegionTypeFromString( value );
    if ( obj->mRegionInfo->mRegionType ==  POMP2_No_type )
    {
        ctcError( obj, CTC_ERROR_Unknown_region_type, value );
    }
}

static int
regionTypesMapCompare( const void* searchKey,
                       const void* mapElem );

static POMP2_Region_type
getRegionTypeFromString( const char* regionTypeString )
{
    RegionTypesMapValueType* mapElem = ( RegionTypesMapValueType* )bsearch(
        regionTypeString,
        &regionTypesMap,
        regionTypesMapSize,
        sizeof( RegionTypesMapValueType ),
        regionTypesMapCompare );

    if ( mapElem )
    {
        return mapElem->mRegionType;
    }
    else
    {
        return POMP2_No_type;
    }
}

static int
regionTypesMapCompare( const void* searchKey,
                       const void* mapElem )
{
    const char* const        key  = ( const char* )searchKey;
    RegionTypesMapValueType* elem = ( RegionTypesMapValueType* )mapElem;

    return strcmp( key, elem->mRegionTypeString );
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
            ctcError( obj, CTC_ERROR_SCL_line_number_error, "line1 > line2" );
        }
    }
    else
    {
        ctcError( obj, CTC_ERROR_SCL_error, 0 );
    }
}

static void
assignHasClause( CTCData*    obj,
                 bool*       hasClause,
                 const char* value )
{
    if ( !isdigit( *value ) )
    {
        ctcError( obj, CTC_ERROR_Wrong_clause_value, value );
    }

    int tmp = atoi( value );
    if ( tmp != 0 && tmp != 1 )
    {
        ctcError( obj, CTC_ERROR_Wrong_clause_value, value );
    }
    *hasClause = tmp;
}

static POMP2_Schedule_type
getScheduleTypeFromString( const char* key );

static void
assignScheduleType( CTCData*       obj,
                    char* restrict value )
{
    char* token = NULL;

    token = strtok( value, "," );

    if ( token )
    {
        obj->mRegionInfo->mScheduleType = getScheduleTypeFromString( token );
    }
    else
    {
        obj->mRegionInfo->mScheduleType = getScheduleTypeFromString( value );
    }
    if ( obj->mRegionInfo->mScheduleType ==  POMP2_No_schedule )
    {
        ctcError( obj, CTC_ERROR_Unknown_schedule_type, value );
    }
}

static void
assignUnsigned( CTCData*    obj,
                unsigned*   anUnsigned,
                const char* value )
{
    int tmp = atoi( value );
    if ( tmp < 0 )
    {
        ctcError( obj, CTC_ERROR_Unsigned_expected, value );
    }
    *anUnsigned = tmp;
}

static void
assignString( char**      aString,
              const char* value )
{
    *aString = malloc( strlen( value ) * sizeof( char ) + 1 );
    strcpy( *aString, value );
}

static void
checkConsistency( CTCData* obj )
{
    bool requiredAttributesFound;

    if ( obj->mRegionInfo->mRegionType == POMP2_No_type )
    {
        ctcError( obj, CTC_ERROR_No_region_type, 0 );
        return;
    }

    requiredAttributesFound = ( obj->mRegionInfo->mStartFileName
                                && obj->mRegionInfo->mEndFileName );
    if ( !requiredAttributesFound )
    {
        ctcError( obj, CTC_ERROR_SCL_broken, 0 );
        return;
    }


    if ( obj->mRegionInfo->mRegionType == POMP2_Sections
         && obj->mRegionInfo->mNumSections <= 0 )
    {
        ctcError( obj, CTC_ERROR_Num_sections_invalid, 0 );
        return;
    }

    if ( obj->mRegionInfo->mRegionType == POMP2_User_region
         && obj->mRegionInfo->mUserRegionName == 0 )
    {
        ctcError( obj, CTC_ERROR_User_region_name_missing, 0 );
        return;
    }

    if ( obj->mRegionInfo->mStartLine1 > obj->mRegionInfo->mStartLine2 )
    {
        ctcError( obj, CTC_ERROR_Inconsistent_line_numbers, 0 );
        return;
    }

    if ( obj->mRegionInfo->mEndLine1 > obj->mRegionInfo->mEndLine2 )
    {
        ctcError( obj, CTC_ERROR_Inconsistent_line_numbers, 0 );
        return;
    }
/* A barrier, taskwait and flush does not have an end line number, since it
 * is not associated to a region.*/
    if ( obj->mRegionInfo->mStartLine2 > obj->mRegionInfo->mEndLine1 &&
         obj->mRegionInfo->mRegionType != POMP2_Barrier &&
         obj->mRegionInfo->mRegionType != POMP2_Taskwait &&
         obj->mRegionInfo->mRegionType != POMP2_Flush )
    {
        ctcError( obj, CTC_ERROR_Inconsistent_line_numbers, 0 );
        return;
    }
}

/*----------------------------------------------------------------------------*/

void
freePOMP2RegionInfoMembers( POMP2_Region_info* regionInfo )
{
    freeAndReset( &regionInfo->mStartFileName );
    freeAndReset( &regionInfo->mEndFileName );
    freeAndReset( &regionInfo->mCriticalName );
    freeAndReset( &regionInfo->mUserRegionName );
    freeAndReset( &regionInfo->mUserGroupName );
}

/*----------------------------------------------------------------------------*/

/** @brief returns a region string*/
const char*
pomp2RegionType2String( POMP2_Region_type regionType )
{
    int i;
    for ( i = 0; i < regionTypesMapSize; ++i )
    {
        if ( regionType == regionTypesMap[ i ].mRegionType )
        {
            return regionTypesMap[ i ].mRegionTypeString;
        }
    }
    return "no valid region type";
}

/*----------------------------------------------------------------------------*/

/** @brief matching between schedule string description and schedule type*/
typedef struct
{
    /** string representation */
    char*               mScheduleTypeString;
    /** matching schedule type*/
    POMP2_Schedule_type mScheduleType;
} ScheduleTypesMapValueType;

/** @brief map with schedule types*/
static const ScheduleTypesMapValueType scheduleTypesMap[] =
{
    /* Entries must be sorted to be used in binary search. */
    /* If you add/remove items, scheduleTypesMapSize       */
    { "auto",    POMP2_Auto    },
    { "dynamic", POMP2_Dynamic },
    { "guided",  POMP2_Guided  },
    { "runtime", POMP2_Runtime },
    { "static",  POMP2_Static  }
};

/** @brief number of entries in scheduleTypesMap*/
const size_t scheduleTypesMapSize = sizeof( scheduleTypesMap ) / sizeof( ScheduleTypesMapValueType );

static int
scheduleTypesMapCompare( const void* searchKey,
                         const void* mapElem );

/* @brief Assigns a schedule type according to an entry in the ctc string */
static POMP2_Schedule_type
getScheduleTypeFromString( const char* key )
{
    ScheduleTypesMapValueType* mapElem = ( ScheduleTypesMapValueType* )bsearch( key,
                                                                                &scheduleTypesMap,
                                                                                scheduleTypesMapSize,
                                                                                sizeof( ScheduleTypesMapValueType ),
                                                                                scheduleTypesMapCompare );

    if ( mapElem )
    {
        return mapElem->mScheduleType;
    }
    else
    {
        return POMP2_No_schedule;
    }
}

static int
scheduleTypesMapCompare( const void* searchKey,
                         const void* mapElem )
{
    const char* const          key  = ( const char* )searchKey;
    ScheduleTypesMapValueType* elem = ( ScheduleTypesMapValueType* )mapElem;

    return strcmp( key, elem->mScheduleTypeString );
}

/** @brief returns a string of the schedule type*/
const char*
pomp2ScheduleType2String( POMP2_Schedule_type scheduleType )
{
    int i;
    for ( i = 0; i < scheduleTypesMapSize; ++i )
    {
        if ( scheduleType == scheduleTypesMap[ i ].mScheduleType )
        {
            return scheduleTypesMap[ i ].mScheduleTypeString;
        }
    }
    return "no valid schedule type";
}
