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
 *  @file       ompregion.cc
 *  @status     beta
 *
 *  @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
 *
 *  @authors    Bernd Mohr <b.mohr@fz-juelich.de>
 *              Dirk Schmidl <schmidl@rz-rwth-aachen.de>
 *              Peter Philippen <p.philippen@fz-juelich.de>

 *  @brief      Functions to handle parallel regions. Including the creation
 *              and initialization of region handles. */

#include <config.h>
#include "ompregion.h"
#include "opari2.h"
//#include <iomanip>
//using std::setw;
//using std::setfill;
#include "common.h"
#include <cassert>
#include "handler.h"

#define MAKE_STR( x ) MAKE_STR_( x )
#define MAKE_STR_( x ) #x

OMPRegion::OMPRegion( const string& n,
                      const string& file,
                      int           bfl,
                      int           bll,
                      bool          outr )
    : name( n ), file_name( file ), id( ++maxId ),
      begin_first_line( bfl ), begin_last_line( bll ),
      end_first_line( 0 ), end_last_line( 0 ),
      num_sections( 0 ), noWaitAdded( false ), has_untied( false ),
      has_if( false ), has_num_threads( false ), has_reduction( false ),
      has_schedule( false ), arg_schedule( "" ), has_collapse( false ),
      has_ordered( false ), outer_reg( outr ), enclosing_reg( 0 )
{
    enclosing_reg = outer_ptr;
    if ( outer_reg )
    {
        outer_ptr = this;
    }
    if ( outer_ptr )
    {
        outer_ptr->descrs.insert( id );
    }
}

OMPRegion::OMPRegion( const OMPRegion& parent,
                      const string&    n,
                      const string&    file,
                      int              bfl,
                      int              bll,
                      bool             outr )
    : name( n ), file_name( file ), id( parent.id ),
      begin_first_line( bfl ), begin_last_line( bll ),
      end_first_line( 0 ), end_last_line( 0 ),
      num_sections( 0 ), noWaitAdded( false ), has_untied( false ),
      has_if( false ), has_num_threads( false ), has_reduction( false ),
      has_schedule( false ), arg_schedule( "" ), has_collapse( false ),
      has_ordered( false ), outer_reg( outr ), enclosing_reg( 0 )
{
    enclosing_reg = outer_ptr;
    if ( outer_reg )
    {
        outer_ptr = this;
    }
    if ( outer_ptr )
    {
        outer_ptr->descrs.insert( id );
    }
}

void
OMPRegion::generate_header_c( ostream& os )
{
    os << "#include <opari2/pomp2_lib.h>\n\n";
    if ( copytpd )
    {
        os << "#include <stdint.h>\n";
        os << "extern int64_t " << MAKE_STR( FORTRAN_ALIGNED ) " " << pomp_tpd << ";\n";
        os << "#pragma omp threadprivate(" << pomp_tpd << ")\n";
    }
}

void
OMPRegion::generate_header_cxx( ostream& os )
{
    os << "#include <opari2/pomp2_lib.h>\n\n";
    if ( copytpd )
    {
        os << "#include <stdint.h>\n";
        os << "extern \"C\" \n{\n";
        os << "extern int64_t " << MAKE_STR( FORTRAN_ALIGNED ) " " << pomp_tpd << ";\n";
        os << "#pragma omp threadprivate(" << pomp_tpd << ")\n";
        os << "}\n";
    }
}
/** @brief Generate a function to allow initialization of all ompregion handles for Fortran.
 *         These functions need to be called from POMP_Init_regions.*/
void
OMPRegion::generate_init_handle_calls_f( ostream& os, const char* incfile )
{
    // I'm not sure, how portable using gettimeofday() is. We rely on a
    // sufficient resolution of timeval.tv_usec.

    if ( OMPRegion::init_handle_calls.rdbuf()->in_avail() != 0 )
    {
        //add a Function to initialize the handles at the end of the file
        os << "\n      subroutine POMP2_Init_regions_"
           << compiletime.tv_sec << compiletime.tv_usec
           << "_" << OMPRegion::maxId << "()\n"
           << "         include \'" << incfile << "\'\n"
           << OMPRegion::init_handle_calls.str()
           << "      end\n";
    }
}

/** @brief Generate a function to allow initialization of all ompregion handles for C++.
 *         The function uses extern "C" to avoid complicate name mangling issues.*/
void
OMPRegion::generate_init_handle_calls_cxx( ostream& os )
{
    // I'm not sure, how portable using gettimeofday() is. We rely on a
    // sufficient resolution of timeval.tv_usec.
    //int     retval = gettimeofday( &compiletime, NULL );
    //assert( retval == 0 );

    os << "extern \"C\" \n{"
       << "\nvoid POMP2_Init_regions_"
       << compiletime.tv_sec << compiletime.tv_usec
       << "_" << OMPRegion::maxId << "()\n{\n"
       << OMPRegion::init_handle_calls.str()
       << "}\n}\n";
}

/** @brief Generate a function to allow initialization of all ompregion handles for C.*/
void
OMPRegion::generate_init_handle_calls_c( ostream& os )
{
    // I'm not sure, how portable using gettimeofday() is. We rely on a
    // sufficient resolution of timeval.tv_usec.
    //int     retval = gettimeofday( &compiletime, NULL );
    //assert( retval == 0 );

    os << "\nvoid POMP2_Init_regions_"
       << compiletime.tv_sec << compiletime.tv_usec
       << "_" << OMPRegion::maxId << "()\n{\n"
       << OMPRegion::init_handle_calls.str()
       << "}\n";
}

void
OMPRegion::finalize_descrs( ostream& os, Language lang )
{
    if ( lang & L_F77 )
    {
        os << "\n      integer*4 pomp2_lib_get_max_threads";
        os << "\n      logical pomp2_test_lock";
        os << "\n      integer*4 pomp2_test_nest_lock\n";
    }
    else
    {
        os << "\n      integer ( kind=4 ) :: pomp2_lib_get_max_threads";
        os << "\n      logical :: pomp2_test_lock";
        os << "\n      integer ( kind=4 ) :: pomp2_test_nest_lock\n";
    }


    if ( !common_block.empty() )
    {
        vector<int>::iterator it = common_block.begin();
        if ( copytpd )
        {
            if ( lang & L_F77 )
            {
                os << "      integer*8 pomp_tpd \n";
            }
            else
            {
                os << "      integer( kind=8 ) pomp_tpd \n";
            }
            os << "      common /pomp_tpd/ pomp_tpd \n";
            os << "!$omp threadprivate(/pomp_tpd/)\n";
        }

        if ( lang & L_F77 )
        {
            os << "      integer*8 pomp2_old_task, pomp2_new_task \n";
            os << "      logical pomp_if \n";
            os << "      integer*4 pomp_num_threads \n";
        }
        else
        {
            os << "      integer ( kind=8 ) :: pomp2_old_task, pomp2_new_task \n";
            os << "      logical :: pomp_if \n";
            os << "      integer ( kind=4 ) :: pomp_num_threads \n";
        }
        os << "      common /" << "cb" << compiletime.tv_sec << compiletime.tv_usec << "/ " << region_id_prefix << *it++;
        for (; it < common_block.end(); it++ )
        {
            if ( lang & L_F77 )
            {
                os << ",\n     &          " << region_id_prefix << *it;
            }
            else
            {
                os << ",&\n              " << region_id_prefix << *it;
            }
        }
        os << std::endl;
    }
}

/** @brief Generates the ctc-string */
string
OMPRegion::generate_ctc_string( Language lang )
{
    stringstream stream1, stream2;
    string       ctc_string;

    stream1 << "*regionType=" << name             << "*"
            << "sscl="  << file_name        << ":"
            << begin_first_line << ":"
            << begin_last_line         << "*"
            << "escl="  << file_name        << ":"
            << end_first_line   << ":"
            << end_last_line           << "*";
    if ( name == "critical" )
    {
        if ( !sub_name.empty() )
        {
            stream1 << "criticalName=" << sub_name << "*";
        }
    }
    else if ( name == "sections" )
    {
        stream1 << "numSections=" << num_sections << "*";
    }
    else if ( name == "region" )
    {
        stream1 << "userRegionName=" << sub_name << "*";
    }

    if ( has_if )
    {
        stream1 << "hasIf=1*";
    }
    if ( has_num_threads )
    {
        stream1 << "hasNumThreads=1*";
    }
    if ( has_reduction )
    {
        stream1 << "hasReduction=1*";
    }
    if ( has_schedule )
    {
        stream1 << "scheduleType=" << arg_schedule << "*";
    }
    if ( has_collapse )
    {
        stream1 << "hasCollapse=1*";
    }
    if ( has_ordered )
    {
        stream1 << "hasOrdered=1*";
    }
    if ( has_untied )
    {
        stream1 << "hasUntied=1*";
    }


    stream1 << "*\"";
    ctc_string = stream1.str();

    stream2 << "\"" << ctc_string.length() - 2 << ctc_string;
    ctc_string = stream2.str();

    if ( lang & L_FORTRAN )
    {
        if ( lang & L_F77 )
        {
            for ( unsigned int i = 58; i < ctc_string.size() - 1; i += 68 )
            {
                ctc_string.insert( i, "\"//\n     &\"" );
            }
        }
        else
        {
            for ( unsigned int i = 58; i < ctc_string.size() - 1; i += 68 )
            {
                ctc_string.insert( i, "\"//&\n      \"" );
            }
        }
    }
    return ctc_string;
}

/** @brief Generate region descriptors for Fortran in the inc file.*/
void
OMPRegion::generate_descr_f( ostream& os, Language lang )
{
    if ( lang & L_F77 )
    {
        os << "      INTEGER*8 " << region_id_prefix << id << "\n";
    }
    else
    {
        os << "      INTEGER( KIND=8 ) :: " << region_id_prefix << id << "\n";
    }
    common_block.push_back( id );

    init_handle_calls << "         call POMP2_Assign_handle( "
                      << region_id_prefix << id << ", ";
    if ( lang & L_F77 )
    {
        init_handle_calls << "\n     &";
    }
    else
    {
        init_handle_calls << "&\n     ";
    }

    init_handle_calls << generate_ctc_string( lang ) << " )\n";
}

/** @brief generate region descriptors in C.*/
void
OMPRegion::generate_descr_c( ostream& os )
{
    if ( !descrs.empty() )
    {
        // Workaround for Suse 11.3 & iomanip -> intel 11.1 compiler bug
        stringstream ids;
        ids << id;

        os << "#define POMP2_DLIST_";
        for ( int i = ids.str().length(); i < 5; i++ )
        {
            os << '0';
        }
        os << ids.str() << " shared(";

        // This can be used again, as soon as the corresponding bug with Suse 11.3 and iomanip is fixed by intel.
        //        os << "#define POMP2_DLIST_" << setw( 5 ) << setfill( '0' ) << id
        //           << " shared(";

        for ( set<int>::const_iterator it = descrs.begin();
              it != descrs.end(); ++it )
        {
            if ( it != descrs.begin() )
            {
                os << ",";
            }
            os << region_id_prefix << *it;
        }
        os << ")\n";
    }

    os << "static POMP2_Region_handle " << region_id_prefix << id << " = NULL;\n";

    init_handle_calls << "    POMP2_Assign_handle( &" << region_id_prefix << id
                      << ", " << generate_ctc_string( L_C ) << " );\n";
}

void
OMPRegion::finish()
{
    if ( outer_reg )
    {
        outer_ptr = enclosing_reg;
    }
}

timeval                 compiletime;
stringstream OMPRegion::init_handle_calls;
vector<int> OMPRegion:: common_block;
OMPRegion* OMPRegion::  outer_ptr = 0;
int OMPRegion::         maxId     = 0;
