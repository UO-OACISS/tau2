/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2013,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2013,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2013,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2013,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2013,
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2013,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2013,
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

/** @internal
 *
 *  @file  opari2.h
 *
 *  @brief This file contains some definitions that are used
 *         throughout OAPRI2
 */


#ifndef OPARI2_H
#define OPARI2_H

#include <string>
using std::string;
#include <fstream>
using std::ifstream;
using std::ofstream;
#include <stdint.h>
#include <map>
using std::map;
#include <vector>
using std::vector;
#include <utility>
using std::pair;


/**
 *  @brief Convenience type definition
 */
typedef map<string, string> OPARI2_StrStr_map_t;

/**
 *  @brief Convenience type definition
 */
typedef map<string, vector<string> > OPARI2_StrVStr_map_t;

/**
 *  @brief Convenience type definition
 */
typedef vector<pair<string, bool> > OPARI2_StrBool_pairs_t;


/**
 *  @brief Convenience type definition
 */
typedef vector<pair<string, string> > OPARI2_StrStr_pairs_t;


/**
 *  @brief Bitfield for supported languages and dialects.
 */
typedef enum
{
    L_NA  = 0x00,
    L_F77 = 0x01, L_F90 = 0x02, L_FORTRAN  = 0x03,
    L_C   = 0x04, L_CXX = 0x08, L_C_OR_CXX = 0x0C
} OPARI2_Language_t;


/**
 *  @brief Fortran specific differentiation between free-form and
 *         fixed-form Fortran format.
 */
typedef enum
{
    F_NA = 0x00, F_FIX = 0x01, F_FREE = 0x02
} OPARI2_Format_t;


/**
 *  @brief All necessary information to process a file.
 */
typedef struct
{
    /** Language of the input file */
    OPARI2_Language_t lang;
    /** Fortran format of the input file */
    OPARI2_Format_t   form;
    /** Specifies whether the source code information should be kept
        up-to-date with line directives*/
    bool keep_src_info;
    /** Specifies whether the input file was already partially
        preprocessed */
    bool     preprocessed_file;
    /** Name of the input file */
    string   infile;
    /** Name of the output file */
    string   outfile;
    /** Name of the generated include file (without path) */
    string   incfile_nopath;
    /** Name of the generated include file (including path) */
    string   incfile;
    /** Input file stream object */
    ifstream is;
    /** Output file stream object */
    ofstream os;
} OPARI2_Option_t;


/**
 *  @brief Definition of directive group.
 */
typedef enum OPARI2_ParadigmType
{
    /** Unknown paradigm */
    OPARI2_PT_NONE = 0,
    /** OpenMP */
    OPARI2_PT_OMP,
    /** POMP user instrumentation */
    OPARI2_PT_POMP,
    /** OpenACC */
    OPARI2_PT_OPENACC,
    /** Intel offload model */
    OPARI2_PT_OFFLOAD,
    /** Transactional memory / speculative execution */
    OPARI2_PT_TMSE
} OPARI2_ParadigmType_t;


/**
 *  @brief Definition of error codes.
 */
typedef enum OPARI2_ErrorCode
{
    OPARI2_NO_ERROR = 0,
    /** No message was printed when the error occurred */
    OPARI2_ERROR_NO_MESSAGE,
    /** A message was printed directly when the error occurred */
    OPARI2_ERROR_WITH_MESSAGE
} OPARI2_ErrorCode;

/**
 *  @brief Definition of null group
 *
 * Currently the max number of groups for each paradigm is 32.
 * This limitation can be removed if using macro definion instead of enum.
 * Note a group is identified by both the group number and the paradigm type.
 */

#define G_NONE 0x00000000

//extern OPARI2_Option_t opt;

/**
 *  @brief This function can be called anywhere upon an error.
 *
 *  When this function is called, the output file is deleted and the
 *  program is exited.
 */
void
cleanup_and_exit( void );


#endif
