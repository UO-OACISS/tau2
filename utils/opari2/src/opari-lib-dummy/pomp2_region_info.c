/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2011,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2011,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2011,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2011,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2011, 2013, 2014
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2011,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2011,
 * Technische Universitaet Muenchen, Germany
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
 * @file    pomp2_region_info.c
 * @date    Started Fri Mar 20 16:59:41 2009
 *
 * @brief
 *
 */

#include <config.h>

#include "opari2_ctc_parser.h"
#include "opari2_ctc_token.h"
#include "pomp2_region_info.h"

#include <assert.h>
#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*----------------------------------------------------------------------------*/

static void
checkOMPConsistency( CTCData* obj );

static void
initOpenmpRegionInfo( POMP2_Region_info* obj );

bool
checkCTCTokenAndAssignRegionInfoValues( int      ctctoken,
                                        char*    value,
                                        CTCData* obj );

void
ctcString2RegionInfo( const char         string[],
                      POMP2_Region_info* regionInfo )
{
    assert( regionInfo );

    CTCData ctcData;

    ctcData.mRegionInfo = ( OPARI2_Region_info* )regionInfo;

    initOpenmpRegionInfo( regionInfo );
    OPARI2_CTC_initCTCData( &ctcData, string );

    OPARI2_CTC_parseCTCStringAndAssignRegionInfoValues
        ( &ctcData, checkCTCTokenAndAssignRegionInfoValues );

    checkOMPConsistency( &ctcData );
    OPARI2_CTC_freeCTCData( &ctcData );
}

static void
initOpenmpRegionInfo( POMP2_Region_info* obj )
{
    obj->mRegionType      = POMP2_No_type;
    obj->mHasCollapse     = false;
    obj->mHasCopyIn       = false;
    obj->mHasCopyPrivate  = false;
    obj->mHasFirstPrivate = false;
    obj->mHasIf           = false;
    obj->mHasLastPrivate  = false;
    obj->mHasNoWait       = false;
    obj->mHasNumThreads   = false;
    obj->mHasOrdered      = false;
    obj->mHasReduction    = false;
    obj->mHasUntied       = false;
    obj->mScheduleType    = POMP2_No_schedule;
    obj->mNumSections     = 0;
    obj->mCriticalName    = 0;
    obj->mUserGroupName   = 0;
}

static void
assignRegionType( CTCData*    obj,
                  const char* value );

static void
assignScheduleType( CTCData*       obj,
                    char* restrict value );

static void
assignDefaultSharingType( CTCData*       obj,
                          char* restrict value );

bool
checkCTCTokenAndAssignRegionInfoValues( int      ctctoken,
                                        char*    value,
                                        CTCData* obj )
{
    POMP2_Region_info* regionInfo = ( POMP2_Region_info* )obj->mRegionInfo;

    switch ( ctctoken )
    {
        case CTC_Region_type:
            assignRegionType( obj, value );
            return true;
        case CTC_OMP_Has_copy_in:
            OPARI2_CTC_assignHasClause( obj, &regionInfo->mHasCopyIn, value );
            return true;
        case CTC_OMP_Has_copy_private:
            OPARI2_CTC_assignHasClause( obj, &regionInfo->mHasCopyPrivate, value );
            return true;
        case CTC_OMP_Has_first_private:
            OPARI2_CTC_assignHasClause( obj, &regionInfo->mHasFirstPrivate, value );
            return true;
        case CTC_OMP_Has_if:
            OPARI2_CTC_assignHasClause( obj, &regionInfo->mHasIf, value );
            return true;
        case CTC_OMP_Has_last_private:
            OPARI2_CTC_assignHasClause( obj, &regionInfo->mHasLastPrivate, value );
            return true;
        case CTC_OMP_Has_no_wait:
            OPARI2_CTC_assignHasClause( obj, &regionInfo->mHasNoWait, value );
            return true;
        case CTC_OMP_Has_num_threads:
            OPARI2_CTC_assignHasClause( obj, &regionInfo->mHasNumThreads, value );
            return true;
        case CTC_OMP_Has_ordered:
            OPARI2_CTC_assignHasClause( obj, &regionInfo->mHasOrdered, value );
            return true;
        case CTC_OMP_Has_reduction:
            OPARI2_CTC_assignHasClause( obj, &regionInfo->mHasReduction, value );
            return true;
        case CTC_OMP_Has_collapse:
            OPARI2_CTC_assignHasClause( obj, &regionInfo->mHasCollapse, value );
            return true;
        case CTC_OMP_Has_untied:
            OPARI2_CTC_assignHasClause(  obj, &regionInfo->mHasUntied, value );
            return true;
        case CTC_OMP_Has_schedule:
            assignScheduleType( obj, value );
            return true;
        case CTC_OMP_Has_defaultSharing:
            assignDefaultSharingType( obj, value );
            return true;
        case CTC_OMP_Has_shared:
            OPARI2_CTC_assignHasClause( obj, &regionInfo->mHasShared, value );
            return true;
        case CTC_OMP_Num_sections:
            OPARI2_CTC_assignUnsigned( obj, &regionInfo->mNumSections, value );
            return true;
        case CTC_OMP_Critical_name:
            OPARI2_CTC_assignString( &regionInfo->mCriticalName, value );
            return true;
        case CTC_OMP_User_group_name:
            OPARI2_CTC_assignString( &regionInfo->mUserGroupName, value );
            return true;
    }

    return false;
}

/*----------------------------------------------------------------------------*/

/** @brief map with region types*/
static const OPARI2_CTCMapType regionTypesMap[] =
{
    { "atomic",            POMP2_Atomic             },
    { "barrier",           POMP2_Barrier            },
    { "critical",          POMP2_Critical           },
    { "do",                POMP2_Do                 },
    { "flush",             POMP2_Flush              },
    { "for",               POMP2_For                },
    { "master",            POMP2_Master             },
    { "ordered",           POMP2_Ordered            },
    { "parallel",          POMP2_Parallel           },
    { "paralleldo",        POMP2_Parallel_do        },
    { "parallelfor",       POMP2_Parallel_for       },
    { "parallelsections",  POMP2_Parallel_sections  },
    { "parallelworkshare", POMP2_Parallel_workshare },
    { "sections",          POMP2_Sections           },
    { "single",            POMP2_Single             },
    { "task",              POMP2_Task               },
    { "taskuntied",        POMP2_Taskuntied         },
    { "taskwait",          POMP2_Taskwait           },
    { "workshare",         POMP2_Workshare          }
};

static void
assignRegionType( CTCData*    obj,
                  const char* value )
{
    POMP2_Region_info* regionInfo = ( POMP2_Region_info* )obj->mRegionInfo;

    regionInfo->mRegionType =
        ( POMP2_Region_type )OPARI2_CTC_string2Enum( regionTypesMap,
                                                     OPARI2_CTC_MAP_SIZE( regionTypesMap ),
                                                     value );

    if ( regionInfo->mRegionType == POMP2_No_type )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Unknown_region_type, value );
    }
}

const char*
pomp2RegionType2String( POMP2_Region_type regionType )
{
    if ( regionType )
    {
        return OPARI2_CTC_enum2String( regionTypesMap,
                                       OPARI2_CTC_MAP_SIZE( regionTypesMap ),
                                       regionType );
    }

    return "no valid region type";
}

/*----------------------------------------------------------------------------*/

/** @brief map with schedule types*/
static const OPARI2_CTCMapType scheduleTypesMap[] =
{
    { "auto",    POMP2_Auto    },
    { "dynamic", POMP2_Dynamic },
    { "guided",  POMP2_Guided  },
    { "runtime", POMP2_Runtime },
    { "static",  POMP2_Static  }
};

/** @brief returns a string of the schedule type*/
const char*
pomp2ScheduleType2String( POMP2_Schedule_type scheduleType )
{
    if ( scheduleType )
    {
        return OPARI2_CTC_enum2String( scheduleTypesMap,
                                       OPARI2_CTC_MAP_SIZE( scheduleTypesMap ),
                                       scheduleType );
    }

    return "no valid schedule type";
}

static void
assignScheduleType( CTCData*       obj,
                    char* restrict value )
{
    char*              token      = NULL;
    POMP2_Region_info* regionInfo = ( POMP2_Region_info* )obj->mRegionInfo;

    token = strtok( value, "," );

    if ( token )
    {
        regionInfo->mScheduleType =
            ( POMP2_Schedule_type )OPARI2_CTC_string2Enum( scheduleTypesMap,
                                                           OPARI2_CTC_MAP_SIZE( scheduleTypesMap ),
                                                           value );
    }
    else
    {
        regionInfo->mScheduleType =
            ( POMP2_Schedule_type )OPARI2_CTC_string2Enum( scheduleTypesMap,
                                                           OPARI2_CTC_MAP_SIZE( scheduleTypesMap ),
                                                           value );
    }
    if ( regionInfo->mScheduleType ==  POMP2_No_schedule )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Unknown_schedule_type, value );
    }
}

/*----------------------------------------------------------------------------*/

/** @brief map with defaultSharing types*/
static const OPARI2_CTCMapType defaultSharingTypesMap[] =
{
    { "none",         POMP2_None         },
    { "shared",       POMP2_Shared       },
    { "private",      POMP2_Private      },
    { "firstprivate", POMP2_Firstprivate }
};

/** @brief returns a string of the defaultSharing type*/
const char*
pomp2DefaultSharingType2String( POMP2_DefaultSharing_type defaultSharingType )
{
    if ( defaultSharingType )
    {
        return OPARI2_CTC_enum2String( defaultSharingTypesMap,
                                       OPARI2_CTC_MAP_SIZE( defaultSharingTypesMap ),
                                       defaultSharingType );
    }

    return "no valid defaultSharing type";
}

static void
assignDefaultSharingType( CTCData*       obj,
                          char* restrict value )
{
    char*              token      = NULL;
    POMP2_Region_info* regionInfo = ( POMP2_Region_info* )obj->mRegionInfo;

    token = strtok( value, "," );

    if ( token )
    {
        regionInfo->mDefaultSharingType =
            ( POMP2_DefaultSharing_type )OPARI2_CTC_string2Enum( defaultSharingTypesMap,
                                                                 OPARI2_CTC_MAP_SIZE( defaultSharingTypesMap ),
                                                                 value );
    }
    else
    {
        regionInfo->mDefaultSharingType =
            ( POMP2_DefaultSharing_type )OPARI2_CTC_string2Enum( defaultSharingTypesMap,
                                                                 OPARI2_CTC_MAP_SIZE( defaultSharingTypesMap ),
                                                                 value );
    }
    if ( regionInfo->mDefaultSharingType ==  POMP2_No_defaultSharing )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Unknown_default_sharing_type, value );
    }
}

/*----------------------------------------------------------------------------*/

static void
checkOMPConsistency( CTCData* obj )
{
    bool               requiredAttributesFound;
    POMP2_Region_info* regionInfo = ( POMP2_Region_info* )obj->mRegionInfo;

    OPARI2_CTC_checkConsistency( obj );

    if ( regionInfo->mRegionType == POMP2_No_type )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_No_region_type, 0 );
        return;
    }

    if ( regionInfo->mRegionType == POMP2_Sections
         && regionInfo->mNumSections <= 0 )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Num_sections_invalid, 0 );
        return;
    }

/* A barrier, taskwait and flush does not have an end line number, since it
 * is not associated to a region.*/
    if ( obj->mRegionInfo->mStartLine2 > obj->mRegionInfo->mEndLine1 &&
         regionInfo->mRegionType != POMP2_Barrier &&
         regionInfo->mRegionType != POMP2_Taskwait &&
         regionInfo->mRegionType != POMP2_Flush )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Inconsistent_line_numbers, 0 );
        return;
    }
}

/*----------------------------------------------------------------------------*/

void
freePOMP2RegionInfoMembers( POMP2_Region_info* regionInfo )
{
    OPARI2_CTC_freeAndReset( &regionInfo->mCriticalName );
    OPARI2_CTC_freeAndReset( &regionInfo->mUserGroupName );
}
