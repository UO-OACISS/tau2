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
#ifndef POMP2_USER_REGION_INFO_H
#define POMP2_USER_REGION_INFO_H

/**
 * @file    pomp2_user_region_info.h
 * @date    Started Tue Apr 1 2014
 *
 * @brief This file contains function declarations and structs which
 * handle informations on user defined regions. POMP2_USER_Region_info
 * is used to store these informations. It can be filled with a
 * ctcString by ctcString2UserRegionInfo().
 *
 */

#include "opari2_region_info.h"

#include <stdbool.h>

/**
 * POMP2_USER_Region_type
 *
 */
typedef enum
{
    POMP2_USER_no_type = 0,
    POMP2_USER_Region
} POMP2_USER_Region_type;

/** converts regionType into a string
 * @param regionType The regionType to be converted.
 * @return string representation of the region type*/
const char*
pomp2UserRegionType2String( POMP2_USER_Region_type regionType );

/**
 *  @brief This struct stores all information on a user defined
 *  region, like the name or corresponding source lines. The function
 *  ctcString2UserRegionInfo() can be used to fill this struct with data
 *  from a ctcString.
 */
typedef struct
{
    /** @name Generic source code information attributes
     */
    /*@{*/
    /** source location info. Needs to be first for the typecasting
        from generic OPARI2_Region_info to work. */
    OPARI2_REGION_INFO
    /** @name Type of the OpenMP region*/
    POMP2_USER_Region_type mRegionType;
    /*@}*/

    /** @name Attributes for user region types
     */
    /*@{*/
    /** name of a user defined region*/
    char* mUserRegionName;
    /*@}*/
} POMP2_USER_Region_info;

/** CTC Tokens */

#define CTC_USER_REGION_TOKENS  \
    CTC_USER_Region_name

#define CTC_USER_REGION_TOKEN_MAP_ENTRIES            \
    { "userRegionName",  CTC_USER_Region_name }


/**
 * ctcString2UserRegionInfo() fills the POMP2_USER_Region_info object with data read
 * from the ctcString. If the ctcString does not comply with the
 * specification, the program aborts with exit code 1.
 *
 * @n Rationale: ctcString2UserRegionInfo() is used during
 * initialization of the measurement system. If an error occurs, it is
 * better to abort than to struggle with undefined behaviour or @e
 * guessing the meaning of the broken string.
 *
 * @note Can be called from multiple threads concurrently, assuming malloc is
 * thread-safe.
 *
 * @note ctcString2UserRegionInfo() will assign memory to the members of @e
 * regionInfo. You are supposed to to release this memory by calling
 * freePOMP2UserRegionInfoMembers().
 *
 * @param ctcString A string in the format
 * "length*key=value*[key=value]*". The length field is parsed but not
 * used by this implementation. Possible values for key are listed in
 * ctcTokenMap. The string must at least contain values for the keys
 * @c regionType, @c sscl and @c escl. Possible values for the key @c
 * regionType are listed in regionTypesMap. The format for @c sscl
 * resp. @c escl values is @c "filename:lineNo1:lineNo2".
 *
 * @param regionInfo must be a valid object
 *
 * @post At least the required attributes (see POMP2_USER_Region_info) are
 * set.
 * @n If @c regionType=userRegion then
 * POMP2_USER_Region_info::mUserRegionName has a value != 0.
 *
 */
void
ctcString2UserRegionInfo( const char              ctcString[],
                          POMP2_USER_Region_info* regionInfo );

/**
 * Free the memory of the regionInfo members.
 * @param regionInfo The regioninfo to be freed.
 */
void
freePOMP2UserRegionInfoMembers( POMP2_USER_Region_info* regionInfo );


#endif /* POMP2_USER_REGION_INFO_H */
