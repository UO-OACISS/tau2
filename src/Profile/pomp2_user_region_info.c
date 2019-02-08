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
 * @file    pomp2_user_region_info.c
 * @date    Started Tue Apr 1 2014
 *
 * @brief
 *
 */

#include <config.h>

#include "opari2_ctc_parser.h"
#include "opari2_ctc_token.h"
#include "pomp2_user_region_info.h"

#include <assert.h>
#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*----------------------------------------------------------------------------*/

static void
checkUSERConsistency( CTCData* obj );

static void
initUserRegionInfo( POMP2_USER_Region_info* obj );

bool
checkCTCTokenAndAssignUserRegionInfoValues( int      ctctoken,
                                            char*    value,
                                            CTCData* obj );

void
ctcString2UserRegionInfo( const char              string[],
                          POMP2_USER_Region_info* regionInfo )
{
    assert( regionInfo );

    CTCData ctcData;

    ctcData.mRegionInfo = ( OPARI2_Region_info* )regionInfo;

    initUserRegionInfo( regionInfo );
    OPARI2_CTC_initCTCData( &ctcData, string );

    OPARI2_CTC_parseCTCStringAndAssignRegionInfoValues
        ( &ctcData, checkCTCTokenAndAssignUserRegionInfoValues );

    checkUSERConsistency( &ctcData );
    OPARI2_CTC_freeCTCData( &ctcData );
}

static void
initUserRegionInfo( POMP2_USER_Region_info* obj )
{
    obj->mRegionType     = ( POMP2_USER_Region_type )POMP2_USER_no_type;
    obj->mUserRegionName = 0;
}

static void
assignUserRegionType( CTCData*    obj,
                      const char* value );

bool
checkCTCTokenAndAssignUserRegionInfoValues( int      ctctoken,
                                            char*    value,
                                            CTCData* obj )
{
    POMP2_USER_Region_info* regionInfo = ( POMP2_USER_Region_info* )obj->mRegionInfo;

    switch ( ctctoken )
    {
        case CTC_Region_type:
            assignUserRegionType( obj, value );
            return true;
        case CTC_USER_Region_name:
            OPARI2_CTC_assignString( &regionInfo->mUserRegionName, value );
            return true;
    }
    return false;
}

/*----------------------------------------------------------------------------*/

/** @brief map with region types (really short but exactly like the
    other paradigms) */
static const OPARI2_CTCMapType userRegionTypesMap[] =
{
    { "userRegion", POMP2_USER_Region }
};

static void
assignUserRegionType( CTCData*    obj,
                      const char* value )
{
    POMP2_USER_Region_info* regionInfo = ( POMP2_USER_Region_info* )obj->mRegionInfo;

    regionInfo->mRegionType =
        ( POMP2_USER_Region_type )OPARI2_CTC_string2Enum( userRegionTypesMap,
                                                          OPARI2_CTC_MAP_SIZE( userRegionTypesMap ),
                                                          value );

    if ( regionInfo->mRegionType == POMP2_No_type )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_Unknown_region_type, value );
    }
}

const char*
pomp2UserRegionType2String( POMP2_USER_Region_type regionType )
{
    if ( regionType )
    {
        return OPARI2_CTC_enum2String( userRegionTypesMap,
                                       OPARI2_CTC_MAP_SIZE( userRegionTypesMap ),
                                       regionType );
    }

    return "no valid region type";
}

/*----------------------------------------------------------------------------*/

static void
checkUSERConsistency( CTCData* obj )
{
    bool                    requiredAttributesFound;
    POMP2_USER_Region_info* regionInfo = ( POMP2_USER_Region_info* )obj->mRegionInfo;

    OPARI2_CTC_checkConsistency( obj );

    if ( regionInfo->mRegionType == POMP2_USER_no_type )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_No_region_type, 0 );
        return;
    }

    if ( regionInfo->mRegionType == POMP2_USER_Region
         && regionInfo->mUserRegionName == 0 )
    {
        OPARI2_CTC_error( obj, CTC_ERROR_User_region_name_missing, 0 );
        return;
    }
}

/*----------------------------------------------------------------------------*/

void
freePOMP2UserRegionInfoMembers( POMP2_USER_Region_info* regionInfo )
{
    OPARI2_CTC_freeAndReset( &regionInfo->mUserRegionName );
}
