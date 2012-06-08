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
/** @internal
 *
 *  @file       ompregion.h
 *  @status     beta
 *
 *  @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
 *
 *  @authors    Bernd Mohr <b.mohr@fz-juelich.de>
 *              Dirk Schmidl <schmidl@rz-rwth-aachen.de>
 *              Peter Philippen <p.philippen@fz-juelich.de>

 *  @brief      Functions to handle parallel regions. Including the creation
 *              and initialization of region handles. */

#ifndef OMPREGION_H
#define OMPREGION_H
#include <sys/time.h>

#include <iostream>
using std::ostream;
#include <set>
using std::set;
#include <string>
using std::string;
#include <sstream>
using std::stringstream;
#include <vector>
using std::vector;

#include "opari2.h"


/** @brief store and manipulate openmp region related data
 *         and generate output based on the data*/
class OMPRegion
{
public:
    /** increase maxId and use this as id */
    OMPRegion( const string& n,
               const string& file,
               int           bfl,
               int           bll,
               bool          outr = false );

    /** don't increase maxId, but use parent's id */
    OMPRegion( const OMPRegion& parent,
               const string&    n,
               const string&    file,
               int              bfl,
               int              bll,
               bool             outr = false );

    /** generate first lines of include file in C*/
    static void
    generate_header_c( ostream& os );

    /** generate first lines of include file in C++*/
    static void
    generate_header_cxx( ostream& os );

    /** generate call POMP2_Init_region_XXX function to initialize
     *  all handles, in the C case*/
    static void
    generate_init_handle_calls_c( ostream& os );

    /** generate call POMP2_Init_region_XXX function to initialize
     *  all handles, in the C++ case*/
    static void
    generate_init_handle_calls_cxx( ostream& os );

    /** generate call POMP2_Init_region_XXX function to initialize
     *  all handles, in the Fortran case*/
    static void
    generate_init_handle_calls_f( ostream&    os,
                                  const char* incfile );

    /** generate a CTC String*/
    string
    generate_ctc_string( Language lang );

    /** write all descriptors*/
    static void
    finalize_descrs( ostream& os,
                     Language lang );

    /** generate descriptor list for C/C++ files*/
    void
    generate_descr_c( ostream& os );

    /** generate descriptor list for Fortran files*/
    void
    generate_descr_f( ostream& os,
                      Language lang );

/** finish parsing this region and return to outer one*/
    void
    finish();

    /** region name*/
    string     name;
    /** file name*/
    string     file_name;
    /** name of named critical sections*/
    string     sub_name;
    /** pragma id */
    int        id;
    /** first line of the beginning statement*/
    int        begin_first_line;
    /** last line of the beginning statement*/
    int        begin_last_line;
    /** first line of the end statement*/
    int        end_first_line;
    /** last line of the end statement*/
    int        end_last_line;
    /** nummber of sections, if it is a sections pragma*/
    int        num_sections;
    /** is a nowait added?*/
    bool       noWaitAdded;
    /** has untied clause */
    bool       has_untied;
    /** has if clause */
    bool       has_if;
    /** has num_threads clause */
    bool       has_num_threads;
    /** has reduction clause */
    bool       has_reduction;
    /** has schedule clause */
    bool       has_schedule;
    /** argument of schedule clause */
    string     arg_schedule;
    /** has collapse clause */
    bool       has_collapse;
    /** has ordered clause */
    bool       has_ordered;
    /**is this an outer region?*/
    bool       outer_reg;
    /** set of relevant region descriptors*/
    set<int>   descrs;
    /** enclosing region*/
    OMPRegion* enclosing_reg;

    /** time value*/
    static timeval time;
    /** calls to all functions to initialize the handles,
     *  these calls are added to a file unique init function
     *  at the end*/
    static stringstream init_handle_calls;
    /** in Fotran a common block is inserted in the output file
     *  with all region ids, the values are collected here and
     *  printed at the end*/
    static vector<int> common_block;
    /** pointer to outer region*/
    static OMPRegion*  outer_ptr;
    /** max ID*/
    static int         maxId;
};

#endif
