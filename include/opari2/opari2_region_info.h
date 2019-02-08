/****************************************************************************
**  SCALASCA    http://www.scalasca.org/                                   **
**  KOJAK       http://www.fz-juelich.de/jsc/kojak/                        **
*****************************************************************************
**  Copyright (c) 1998-2009                                                **
**  Forschungszentrum Juelich, Juelich Supercomputing Centre               **
**                                                                         **
**  See the file COPYRIGHT in the package base directory for details       **
****************************************************************************/
#ifndef OPARI2_REGION_INFO_H
#define OPARI2_REGION_INFO_H

/**
 * @file    opari2_region_info.h
 * @date    Started Tue Mar 25 2014
 *
 * @brief
 *
 */

#define OPARI2_REGION_INFO                                                 \
    /** name of the corresponding source file from the opening pragma */   \
    char*    mStartFileName;                                               \
    /** line number of the first line from the opening pragma */           \
    unsigned mStartLine1;                                                  \
    /** line number of the last line from the opening pragma */            \
    unsigned mStartLine2;                                                  \
    /** name of the corresponding source file from the closing pragma */   \
    char* mEndFileName;                                                 \
    /** line number of the first line from the closing pragma */           \
    unsigned mEndLine1;                                                    \
    /** line number of the last line from the closing pragma */            \
    unsigned mEndLine2;

#define CTC_REGION_TOKENS           \
    CTC_End_source_code_location,   \
    CTC_Start_source_code_location, \
    CTC_Region_type

#define CTC_REGION_TOKEN_MAP_ENTRIES            \
    { "escl", CTC_End_source_code_location },   \
    { "sscl", CTC_Start_source_code_location }, \
    { "regionType", CTC_Region_type }

/**
 *  @brief This struct stores all information on OPARI2 regions
 */
typedef struct
{
    OPARI2_REGION_INFO
} OPARI2_Region_info;

#endif /* OPARI2_REGION_INFO_H */
