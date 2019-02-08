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
 * Copyright (c) 2009-2013, 2016-2017,
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
 *  @file       opari2_omp_handler.cc
 *
 *  @brief      This file contains all handler funtions to instrument and print
 *              OpenMP pragmas.
 *
 *  @todo A lot of functionality here might be generalized into a
 *        separate writer class. This is planned for the future.
 */

#include <config.h>
#include <vector>
using std::vector;
#include <map>
using std::map;
#include <stack>
using std::stack;
#include <iostream>
using std::cerr;
#include <fstream>
using std::ifstream;
using std::ofstream;
#include <string>
using std::string;
using std::getline;
#include <cstdlib>
using std::exit;
#include <cstring>
using std::strcmp;
#include <cctype>
using std::toupper;
#include <iostream>
using std::ostream;
#include <assert.h>

#include "common.h"
#include "opari2.h"
#include "opari2_directive.h"
#include "opari2_directive_openmp.h"
#include "opari2_directive_manager.h"

extern OPARI2_Option_t opt;

namespace
{
bool in_workshare = false;

void
generate_num_threads( ostream&                os,
                      OPARI2_DirectiveOpenmp* d )
{
    string num_threads = d->GetClauseArg( "num_threads" );

    if ( opt.lang & L_FORTRAN )
    {
        if ( num_threads.length() )
        {
            os << "      pomp2_num_threads = " << num_threads << "\n";
        }
        else
        {
            os << "      pomp2_num_threads = pomp2_lib_get_max_threads()\n";
        }
    }
    else
    {
        if ( num_threads.length() )
        {
            os << "  int pomp2_num_threads = " << num_threads << ";\n";
        }
        else
        {
            os << "  int pomp2_num_threads = omp_get_max_threads();\n";
        }
    }
}

void
generate_if( ostream&                os,
             OPARI2_DirectiveOpenmp* d )
{
    string if_clause = d->GetClauseArg( "if" );

    if ( opt.lang & L_FORTRAN )
    {
        if ( if_clause.length() )
        {
            os << "      pomp2_if = ( " << if_clause << " )\n";
        }
        else
        {
            os << "      pomp2_if = .true.\n";
        }
    }
    else
    {
        if ( if_clause.length() )
        {
            os << "  int pomp2_if = (int)( " << if_clause << " );\n";
        }
        else
        {
            os << "  int pomp2_if = 1;\n";
        }
    }
}

void
generate_call( const char*             event,
               const char*             type,
               int                     id,
               ostream&                os,
               OPARI2_DirectiveOpenmp* d )
{
    char   c1 = toupper( type[ 0 ] );
    string ctc_string;

    if ( opt.lang & L_FORTRAN )
    {
        if ( strcmp( type, "task" ) == 0 || strcmp( type, "untied_task" ) == 0 )
        {
            os << "      if (pomp2_if) then\n";
        }
        os << "      call POMP2_" << c1 << ( type + 1 )
           << "_" << event << "(" << region_id_prefix << id;
        if ( strstr( type, "task" ) != NULL &&
             strcmp( type, "taskwait" ) != 0 &&
             strcmp( event, "begin" ) == 0 )
        {
            os << ", pomp2_new_task";
        }

        if ( d != NULL )
        {
            if ( opt.form == F_FIX )
            {
                os << ",\n     &" << d->GetCTCStringVariable() << " ";
            }
            else
            {
                os << ", &\n     " << d->GetCTCStringVariable() << " ";
            }
        }
        os << ")\n";
        if ( strcmp( type, "task" ) == 0 || strcmp( type, "untied_task" ) == 0 )
        {
            os << "      end if\n";
        }
    }
    else
    {
        if ( strcmp( event, "begin" ) == 0  || strcmp( event, "fork" ) == 0 || strcmp( event, "enter" ) == 0 )
        {
            os << "{ ";
        }

        if ( strcmp( type, "task" ) == 0 || strcmp( type, "untied_task" ) == 0 )
        {
            os << "if (pomp2_if)";
        }

        os << "  POMP2_" << c1 << ( type + 1 )
           << "_" << event << "( &" << region_id_prefix << id;

        if ( strstr( type, "task" ) != NULL &&
             strcmp( type, "taskwait" ) != 0 &&
             strcmp( event, "begin" ) == 0 )
        {
            os << ", pomp2_new_task";
        }

        if ( d != NULL )
        {
            os << ", " << d->GetCTCStringVariable() << " ";
        }
        os << " );";
        if ( strcmp( event, "end" ) == 0 )
        {
            os << " }";
        }
        os << "\n";
        if ( strcmp( event, "join" ) == 0 || strcmp( event, "exit" )  == 0 )
        {
            os << " }\n";
        }
    }
}

void
generate_call_save_task_id( const char*             event,
                            const char*             type,
                            int                     id,
                            ostream&                os,
                            OPARI2_DirectiveOpenmp* d )
{
    char   c1 = toupper( type[ 0 ] );
    string ctc_string;

    if ( opt.lang & L_FORTRAN )
    {
        if ( strcmp( type, "task_create" ) == 0 || strcmp( type, "untied_task_create" ) == 0 )
        {
            os << "      if (pomp2_if) then\n";
        }
        os << "      call POMP2_" << c1 << ( type + 1 )
           << "_" << event << "(" << region_id_prefix << id;
        if ( ( strcmp( type, "task_create" ) == 0 ) || ( strcmp( type, "untied_task_create" ) == 0 )  )
        {
            if ( opt.form == F_FIX )
            {
                os << ",\n     &pomp2_new_task";
            }
            else
            {
                os << ", pomp2_new_task";
            }
        }
        if ( opt.form == F_FIX )
        {
            os << ",\n     &pomp2_old_task";
        }
        else
        {
            os << ",&\n      pomp2_old_task";
        }
        if ( d != NULL )
        {
            if ( opt.form == F_FIX )
            {
                if ( d->GetName() == "task" )
                {
                    os << ", \n     &pomp2_if";
                }
                os << ",\n     &" << d->GetCTCStringVariable() << " ";
            }
            else
            {
                if ( d->GetName() == "task" )
                {
                    os << ", pomp2_if";
                }
                os << ", " << d->GetCTCStringVariable() << " ";
            }
        }
        os << ")\n";
        if ( strcmp( type, "task_create" ) == 0 || strcmp( type, "untied_task_create" ) == 0 )
        {
            os << "      end if\n";
        }
    }
    else
    {
        os << "{ POMP2_Task_handle pomp2_old_task;\n";
        if ( ( strcmp( type, "task_create" ) == 0 ) || ( strcmp( type, "untied_task_create" ) == 0 )  )
        {
            os << "  POMP2_Task_handle pomp2_new_task;\n";
        }
        if ( strcmp( type, "task_create" ) == 0 || strcmp( type, "untied_task_create" ) == 0 )
        {
            os << "if (pomp2_if)";
        }
        os << "  POMP2_" << c1 << ( type + 1 )
           << "_" << event << "( &" << region_id_prefix << id;
        if ( ( strcmp( type, "task_create" ) == 0 ) || ( strcmp( type, "untied_task_create" ) == 0 )  )
        {
            os << ", &pomp2_new_task";
        }
        os << ", &pomp2_old_task";
        if ( d != NULL )
        {
            if ( d->GetName() == "task" )
            {
                os << ", pomp2_if";
            }
            os << ", " << d->GetCTCStringVariable() << " ";
        }
        os << " );\n";
    }
}

void
generate_call_restore_task_id( const char* event,
                               const char* type,
                               int         id,
                               ostream&    os )
{
    char   c1 = toupper( type[ 0 ] );
    string ctc_string;

    if ( opt.lang & L_FORTRAN )
    {
        if ( strcmp( type, "task_create" ) == 0 || strcmp( type, "untied_task_create" ) == 0 )
        {
            os << "      if (pomp2_if) then\n";
        }
        os << "      call POMP2_" << c1 << ( type + 1 )
           << "_" << event << "(" << region_id_prefix << id;
        if ( opt.form == F_FIX )
        {
            os << ",\n     &pomp2_old_task)\n";
        }
        else
        {
            os << ", pomp2_old_task)\n";
        }
        if ( strcmp( type, "task_create" ) == 0 || strcmp( type, "untied_task_create" ) == 0 )
        {
            os << "      end if\n";
        }
    }
    else
    {
        if ( strcmp( type, "task_create" ) == 0 || strcmp( type, "untied_task_create" ) == 0 )
        {
            os << "if (pomp2_if)";
        }
        os << "  POMP2_" << c1 << ( type + 1 )
           << "_" << event << "( &" << region_id_prefix << id;
        os << ", pomp2_old_task ); }\n";
    }
}

/** @brief Instrument an OpenMP Fork.
 *
 * The pomp2_num_threads variable is
 * used to pass the number of requested threads for the
 * parallel region to the POMP library. It is either the
 * result of omp_get_max_threads() or of the num_threads()
 * clause, if present.
 */
void
generate_fork_call( const char*             event,
                    const char*             type,
                    int                     id,
                    ostream&                os,
                    OPARI2_DirectiveOpenmp* d )
{
    char c1 = toupper( type[ 0 ] );

    if ( opt.lang & L_C_OR_CXX )
    {
        os << "{\n";
    }
    generate_num_threads( os, d );
    generate_if( os, d );

    if ( opt.lang & L_FORTRAN )
    {
        os << "      call POMP2_" << c1 << ( type + 1 )
           << "_" << event << "(" << region_id_prefix << id;
        if ( opt.form == F_FIX )
        {
            os << ",\n     &pomp2_if, pomp2_num_threads, pomp2_old_task";
            if ( d != NULL )
            {
                os << ",\n     &" << d->GetCTCStringVariable() << " ";
            }
        }
        else
        {
            os << ",&\n      pomp2_if, pomp2_num_threads, pomp2_old_task";
            if ( d != NULL )
            {
                os << ", &\n      " << d->GetCTCStringVariable() << " ";
            }
        }
        os << ")\n";
    }
    else
    {
        os << "  POMP2_Task_handle pomp2_old_task;\n";
        os << "  POMP2_" << c1 << ( type + 1 ) << "_" << event
           << "(&" << region_id_prefix << id << ", pomp2_if, pomp2_num_threads, "
           << "&pomp2_old_task";
        if ( d != NULL )
        {
            os << ", " << d->GetCTCStringVariable() << " ";
        }
        os << ");\n";
    }
}

/** @brief Generate the OpenMP pragma/directive. */
void
generate_directive( const char* p,
                    int         lineno,
                    const char* filename,
                    ostream&    os )
{
    if ( lineno && opt.keep_src_info )
    {
        // print original source location information reset pragma
        os << "#line " << lineno << " \"" << filename << "\"" << "\n";
    }

    if ( opt.lang & L_FORTRAN )
    {
        os << "!$omp " << p << "\n";
    }
    else
    {
        os << "#pragma omp " << p << "\n";
    }
}

void
generate_barrier( int         n,
                  ostream&    os,
                  const char* filename )
{
    generate_call_save_task_id( "enter", "implicit_barrier", n, os, NULL );
    generate_directive( "barrier", 0, filename, os );
    generate_call_restore_task_id( "exit", "implicit_barrier", n, os );
}
} //end-of-namespace


OPARI2_DirectiveOpenmp*
cast2omp( OPARI2_Directive* d_base )
{
    OPARI2_DirectiveOpenmp* d = dynamic_cast<OPARI2_DirectiveOpenmp*>( d_base );

    if ( d == NULL )
    {
        cerr << "INTERNAL ERROR: OPARI2_DirectiveOpenmp* expected. Please contact user support.\n";
        cleanup_and_exit();
    }

    return d;
}

/**
 * @brief Print the directive's lines, together with additional
 *                statements to support OpenMP task.
 */
void
print_directive_parallel( OPARI2_DirectiveOpenmp* d,
                          ostream&                os )
{
    opari2_omp_option* omp_opt = OPARI2_DirectiveOpenmp::GetOpenmpOpt();
    stringstream       adds;

    if ( opt.lang & L_FORTRAN )
    {
        if ( opt.form == F_FIX )     // fix source form
        {
            adds << "\n!$omp& firstprivate(pomp2_old_task) private(pomp2_new_task)\n!$omp&";
            if ( d->HasClause( "if" ) )
            {
                adds << " if(pomp2_if)";
            }
            adds << " num_threads(pomp2_num_threads)";
            if ( omp_opt->copytpd )
            {
                adds << " copyin(" << omp_opt->pomp_tpd << ")";
            }
            if ( d->ChangedDefault() )
            {
                adds << "\n!$omp& shared(/" << "cb"
                     << d->GetInodeCompiletimeID() << "/)\n"
                     << "!$omp& private(pomp2_if,pomp2_num_threads)";
            }
        }
        else     // free source form
        {
            adds << " &\n  !$omp firstprivate(pomp2_old_task) private(pomp2_new_task) &\n";
            adds << "  !$omp";
            if ( d->HasClause( "if" ) )
            {
                adds << " if(pomp2_if)";
            }
            adds << " num_threads(pomp2_num_threads)";
            if ( omp_opt->copytpd )
            {
                adds << " copyin(" << omp_opt->pomp_tpd << ")";
            }
            if ( d->ChangedDefault() )
            {
                adds << " &\n  !$omp shared(/" << "cb"
                     << d->GetInodeCompiletimeID() << "/)"
                     <<  " &\n  !$omp private(pomp2_if,pomp2_num_threads)";
            }
        }
    }
    else     //C/C++
    {
        adds << " firstprivate(pomp2_old_task)";
        if ( omp_opt->copytpd )
        {
            if ( d->HasClause( "if" ) )
            {
                adds << " if(pomp2_if)";
            }
            adds << " num_threads(pomp2_num_threads) copyin(" << omp_opt->pomp_tpd << ")";
        }
        else
        {
            if ( d->HasClause( "if" ) )
            {
                adds << " if(pomp2_if)";
            }
            adds << " num_threads(pomp2_num_threads)";
        }
    }

    /*  Parallel directives must always be instrumented to enable
     *  a measurement system to do its memory management in a
     *  threadsafe manner. Therefore the line directive must be
     *  inserted even if instrumentation is turned off. */
    if ( opt.keep_src_info && !InstrumentationDisabled( D_FULL ) )
    {
        // print original source location information reset pragma
        os << "#line " << d->GetLineno() << " \"" << d->GetFilename() << "\"" << "\n";
    }

    d->PrintPlainDirective( os, adds.str() );
}

void
enter_handler_notransform( OPARI2_DirectiveOpenmp* d,
                           ostream&                os )
{
    DirectiveStackPush( d );
    d->PrintPlainDirective( os );
}

void
exit_handler_notransform( OPARI2_DirectiveOpenmp* d )
{
    DirectiveStackPop();
}

/**
 * @brief OpenMP pragma transformation functions.
 *
 * These functions are called by directive manager ONLY if the directive is enabled.
 */


/**
 * Note: parallel directive MUST ignore "pomp noinstrument"
 * to ensure the measurement library is thread-safe.
 */
void
h_omp_parallel( OPARI2_Directive* d_base,
                ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_FULL ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion( true );
        int id = d->GetID();
        d->AddDescr( id );

        generate_fork_call( "fork", "parallel", id, os, d );
        print_directive_parallel( d, os );
        generate_call( "begin", "parallel", id, os, NULL );

        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_end_omp_parallel( OPARI2_Directive* d_base,
                    ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_FULL ) )
    {
        d->PrintPlainDirective( os );
        exit_handler_notransform( d );
    }
    else
    {
        int id =  d->ExitRegion( true );
        if ( !InstrumentationDisabled( D_USER ) && DirectiveActive( OPARI2_PT_OMP, "barrier" ) )
        {
            generate_barrier( id, os, d->GetFilename().c_str() );
        }
        generate_call( "end", "parallel", id, os, NULL );
        d->PrintDirective( os );
        generate_call_restore_task_id( "join", "parallel", id, os );
        if ( opt.keep_src_info && !InstrumentationDisabled( D_FULL ) )
        {
            d->ResetSourceInfo( os );
        }
    }
}


void
h_omp_for( OPARI2_Directive* d_base,
           ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion();

        if ( !d->HasClause( "nowait" ) )
        {
            d->AddNowait();
        }
        generate_call( "enter", "for", d->GetID(), os, d );
        d->PrintDirective( os );
    }
}

void
h_end_omp_for( OPARI2_Directive* d_base,
               ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        exit_handler_notransform( d );
    }
    else
    {
        OPARI2_DirectiveOpenmp* d_top = cast2omp( DirectiveStackTop( d ) );
        int                     id    = d->ExitRegion( false );

        if ( d_top->IsNowaitAdded() )
        {
            generate_barrier( id, os, d->GetFilename().c_str() );
        }
        generate_call( "exit", "for", id, os, NULL );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_omp_do( OPARI2_Directive* d_base,
          ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion();

        generate_call( "enter", "do", d->GetID(), os, d );
        d->PrintDirective( os );
    }
}

void
h_end_omp_do( OPARI2_Directive* d_base,
              ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        d->PrintPlainDirective( os );
        exit_handler_notransform( d );
    }
    else
    {
        int id = d->ExitRegion( false );
        if ( d->HasClause( "nowait" ) )
        {
            d->PrintDirective( os );
        }
        else
        {
            d->AddNowait();
            d->PrintDirective( os );
            generate_barrier( id, os, d->GetFilename().c_str() );
        }
        generate_call( "exit", "do", id, os, NULL );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_omp_sections_c( OPARI2_Directive* d_base,
                  ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion();

        if ( !d->HasClause( "nowait" ) )
        {
            d->AddNowait();
        }
        generate_call( "enter", "sections", d->GetID(), os, d );
        d->PrintDirective( os );
    }
}

void
h_omp_section_c( OPARI2_Directive* d_base,
                 ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        OPARI2_DirectiveOpenmp* d_top = cast2omp( DirectiveStackTop( d ) );
        //Init region using the parent directive's region ID!
        d->InitRegion( d_top );

        DirectiveStackPush( d );
        int num_sections = d_top->GetNumSections();

        if ( num_sections )
        {
            d->PrintPlainDirective( os );
        }
        else
        {
            d->PrintDirective( os );
        }
        generate_call( "begin", "section", d_top->GetID(), os, d_top );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }

        num_sections += 1;
        d_top->SetNumSections( num_sections );
    }
}

void
h_end_omp_section_c( OPARI2_Directive* d_base,
                     ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        exit_handler_notransform( d );
    }
    else
    {
        OPARI2_DirectiveOpenmp* d_top = cast2omp( DirectiveStackTop( d ) );
        generate_call( "end", "section", d_top->GetID(), os, NULL );
        DirectiveStackPop();

        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}


void
h_end_omp_section( OPARI2_Directive* d_base,
                   ostream&          os )
{
    if ( opt.lang & L_C_OR_CXX )
    {
        h_end_omp_section_c( d_base, os );
    }
}

void
h_end_omp_sections_c( OPARI2_Directive* d_base,
                      ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        exit_handler_notransform( d );
    }
    else
    {
        OPARI2_DirectiveOpenmp* d_top = cast2omp( DirectiveStackTop( d ) );
        int                     id    = d->ExitRegion( false );

        if ( d_top->IsNowaitAdded() )
        {
            generate_barrier( id, os, d->GetFilename().c_str() );
        }
        generate_call( "exit", "sections", id, os, NULL );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_omp_sections_f( OPARI2_Directive* d_base,
                  ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion();
        generate_call( "enter", "sections", d->GetID(), os, d );
        d->PrintDirective( os );
    }
}

void
h_omp_sections( OPARI2_Directive* d_base,
                ostream&          os )
{
    if ( opt.lang & L_FORTRAN )
    {
        h_omp_sections_f( d_base, os );
    }
    else
    {
        h_omp_sections_c( d_base, os );
    }
}

void
h_omp_section_f( OPARI2_Directive* d_base,
                 ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        d->PrintPlainDirective( os );
    }
    else
    {
        OPARI2_DirectiveOpenmp* d_top        = cast2omp( DirectiveStackTop( d ) );
        int                     num_sections = d_top->GetNumSections();

        if ( num_sections )
        {
            // close last section if necessary
            generate_call( "end", "section", d_top->GetID(), os, NULL );
        }
        d->PrintDirective( os );
        generate_call( "begin", "section", d_top->GetID(), os, d_top );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }

        d_top->SetNumSections( ++num_sections );
    }
}

void
h_omp_section( OPARI2_Directive* d_base,
               ostream&          os )
{
    if ( opt.lang & L_FORTRAN )
    {
        h_omp_section_f( d_base, os );
    }
    else
    {
        h_omp_section_c( d_base, os );
    }
}

void
h_end_omp_sections_f( OPARI2_Directive* d_base,
                      ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        d->PrintPlainDirective( os );
        exit_handler_notransform( d );
    }
    else
    {
        int id = d->ExitRegion( false );
        generate_call( "end", "section", id, os, NULL );
        if ( d->HasClause( "nowait" ) )
        {
            d->PrintDirective( os );
        }
        else
        {
            d->AddNowait();
            d->PrintDirective( os );
            generate_barrier( id, os, d->GetFilename().c_str() );
        }
        generate_call( "exit", "sections", id, os, NULL );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_end_omp_sections( OPARI2_Directive* d_base,
                    ostream&          os )
{
    if ( opt.lang & L_FORTRAN )
    {
        h_end_omp_sections_f( d_base, os );
    }
    else
    {
        h_end_omp_sections_c( d_base, os );
    }
}

void
h_omp_single_c( OPARI2_Directive* d_base,
                ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion();

        if ( !d->HasClause( "nowait" ) && !d->HasClause( "copyprivate" ) )
        {
            d->AddNowait();
        }

        int id = d->GetID();
        generate_call( "enter", "single", id, os, d );
        d->PrintDirective( os );
        generate_call( "begin", "single", id, os, NULL );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_end_omp_single_c( OPARI2_Directive* d_base,
                    ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        exit_handler_notransform( d );
    }
    else
    {
        OPARI2_DirectiveOpenmp* d_top = cast2omp( DirectiveStackTop( d ) );
        int                     id    = d->ExitRegion( false );

        generate_call( "end", "single", id, os, NULL );
        if ( d_top->IsNowaitAdded() )
        {
            generate_barrier( id, os, d->GetFilename().c_str() );
        }
        generate_call( "exit", "single", id, os, NULL );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_omp_single_f( OPARI2_Directive* d_base,
                ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion();

        int id = d->GetID();
        generate_call( "enter", "single", id, os, d );
        d->PrintDirective( os );
        generate_call( "begin", "single", id, os, NULL );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_omp_single( OPARI2_Directive* d_base,
              ostream&          os )
{
    if ( opt.lang & L_FORTRAN )
    {
        h_omp_single_f( d_base, os );
    }
    else
    {
        h_omp_single_c( d_base, os );
    }
}

void
h_end_omp_single_f( OPARI2_Directive* d_base,
                    ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        d->PrintPlainDirective( os );
        exit_handler_notransform( d );
    }
    else
    {
        int id = d->ExitRegion( false );
        generate_call( "end", "single", id, os, NULL );
        if ( d->HasClause( "nowait" ) )
        {
            d->PrintDirective( os );
        }
        else
        {
            if ( !d->HasClause( "copyprivate" ) )
            {
                d->AddNowait();
            }
            d->PrintDirective( os );
            if ( d->IsNowaitAdded() )
            {
                generate_barrier( id, os, d->GetFilename().c_str() );
            }
        }
        generate_call( "exit", "single", id, os, NULL );

        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_end_omp_single( OPARI2_Directive* d_base,
                  ostream&          os  )
{
    if ( opt.lang & L_FORTRAN )
    {
        h_end_omp_single_f( d_base, os );
    }
    else
    {
        h_end_omp_single_c( d_base, os );
    }
}


void
h_omp_master( OPARI2_Directive* d_base,
              ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion();

        d->PrintDirective( os );
        generate_call( "begin", "master", d->GetID(), os, d );

        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_end_omp_master_c( OPARI2_Directive* d_base,
                    ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        exit_handler_notransform( d );
    }
    else
    {
        int id = d->ExitRegion( false );

        generate_call( "end", "master", id, os, NULL );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_end_omp_master_f( OPARI2_Directive* d_base,
                    ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        d->PrintPlainDirective( os );
        exit_handler_notransform( d );
    }
    else
    {
        int id = d->ExitRegion( false );

        generate_call( "end", "master", id, os, NULL );
        d->PrintDirective( os );
    }
}

void
h_end_omp_master( OPARI2_Directive* d_base,
                  ostream&          os )
{
    if ( opt.lang & L_FORTRAN )
    {
        h_end_omp_master_f( d_base, os );
    }
    else
    {
        h_end_omp_master_c( d_base, os );
    }
}

void
h_omp_critical( OPARI2_Directive* d_base,
                ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    /** 'critical' should NOT be instrumented if inside 'workshare' construct */
    if ( InstrumentationDisabled( D_USER ) || in_workshare )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion();

        int id = d->GetID();
        generate_call( "enter", "critical", id, os, d );
        d->PrintDirective( os );

        generate_call( "begin", "critical", id, os, NULL );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_end_omp_critical( OPARI2_Directive* d_base,
                    ostream&          os )
{
    OPARI2_DirectiveOpenmp* d     = cast2omp( d_base );
    OPARI2_DirectiveOpenmp* d_top = cast2omp( DirectiveStackTop( d ) );

    if ( InstrumentationDisabled( D_USER ) || in_workshare )
    {
        d->PrintPlainDirective( os );
        exit_handler_notransform( d );
    }
    else
    {
        int     id   = d->ExitRegion( false );
        string& name = d->GetName();

        string& subname     = d->GetUserName();
        string& subname_top = d_top->GetUserName();

        if ( name[ 0 ] != '$' )
        {
            if ( subname != subname_top )
            {
                cerr << d->GetFilename() << ":" << d_top->GetLineno()
                     << ": ERROR: missing end critical(" << subname
                     << ") directive\n";
                cerr << d->GetFilename() << ":" << d->GetLineno()
                     << ": ERROR: non-matching end critical(" << subname
                     << ") directive\n";
                cleanup_and_exit();
            }
        }

        generate_call( "end", "critical", id, os, NULL );
        d->PrintDirective( os );
        generate_call( "exit", "critical", id, os, NULL );

        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_omp_parallelfor( OPARI2_Directive* d_base,
                   ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );
    if ( InstrumentationDisabled( D_FULL ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion( true );

        int id = d->GetID();
        d->AddDescr( id );

        OPARI2_DirectiveOpenmp* for_directive = d->SplitCombined();

        if ( InstrumentationDisabled( D_USER ) || !( d->active ) )
        {
            generate_fork_call( "fork", "parallel", id, os, d );
            print_directive_parallel( d, os );
            generate_call( "begin", "parallel", id, os, NULL );
        }
        else
        {
            for_directive->AddNowait();
            generate_fork_call( "fork", "parallel", id, os, d );
            print_directive_parallel( d, os );
            generate_call( "begin", "parallel", id, os, NULL );
            generate_call( "enter", "for", id, os, d );
        }


        for_directive->PrintDirective( os ); // #omp for nowait
        delete for_directive;
    }
}

void
h_end_omp_parallelfor( OPARI2_Directive* d_base,
                       ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_FULL ) )
    {
        exit_handler_notransform( d );
    }
    else
    {
        int id = d->ExitRegion( true );


        if ( !InstrumentationDisabled( D_USER ) && ( d->active ) )
        {
            generate_barrier( id, os, d->GetFilename().c_str() );
            generate_call( "exit", "for", id, os, NULL );
        }

        generate_call( "end", "parallel", id, os, NULL );
        generate_call_restore_task_id( "join", "parallel", id, os );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_omp_paralleldo( OPARI2_Directive* d_base,
                  ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_FULL ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion( true );

        int id = d->GetID();
        d->AddDescr( id );

        generate_fork_call( "fork", "parallel", id, os, d );

        OPARI2_DirectiveOpenmp* do_directive = d->SplitCombined();

        print_directive_parallel( d, os );
        generate_call( "begin", "parallel", id, os, NULL );

        if ( !InstrumentationDisabled( D_USER )  && ( d->active ) )
        {
            generate_call( "enter", "do", id, os, d );
        }

        do_directive->PrintDirective( os );  // #omp do

        delete do_directive;
    }
}

void
h_end_omp_paralleldo( OPARI2_Directive* d_base,
                      ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_FULL ) )
    {
        d->PrintPlainDirective( os );
        exit_handler_notransform( d );
    }
    else
    {
        int     id       = d->ExitRegion( true );
        string& filename = d->GetFilename();
        int     lineno   = d->GetLineno();

        if ( InstrumentationDisabled( D_USER ) || !( d->active ) )
        {
            generate_directive( "end do", lineno, filename.c_str(), os );
        }
        else
        {
            generate_directive( "end do nowait", lineno, filename.c_str(), os );
            generate_barrier( id, os, filename.c_str() );
            generate_call( "exit", "do", id, os, NULL );
        }


        generate_call( "end", "parallel", id, os, NULL );
        generate_directive( "end parallel", lineno, filename.c_str(), os );
        generate_call_restore_task_id( "join", "parallel", id, os );

        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_omp_parallelsections_c( OPARI2_Directive* d_base,
                          ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_FULL ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion( true );

        int id = d->GetID();
        d->AddDescr( id );

        OPARI2_DirectiveOpenmp* d_section = d->SplitCombined();
        if ( !InstrumentationDisabled( D_USER ) && ( d->active ) )
        {
            d_section->AddNowait();
        }

        generate_fork_call( "fork", "parallel", id, os, d );
        print_directive_parallel( d, os );
        generate_call( "begin", "parallel", id, os, NULL );

        if ( !InstrumentationDisabled( D_USER ) && ( d->active ) )
        {
            generate_call( "enter", "sections", id, os, d );
        }

        d_section->PrintDirective( os ); // #omp sections

        delete d_section;
    }
}

void
h_omp_parallelsections_f( OPARI2_Directive* d_base,
                          ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_FULL ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion( true );

        int id = d->GetID();
        d->AddDescr( id );

        OPARI2_DirectiveOpenmp* sec_directive = d->SplitCombined();
        generate_fork_call( "fork", "parallel", id, os, d );
        print_directive_parallel( d, os );        // #omp parallel
        generate_call( "begin", "parallel", id, os, NULL );

        if ( !InstrumentationDisabled( D_USER ) && ( d->active ) )
        {
            generate_call( "enter", "sections", id, os, NULL );
        }

        sec_directive->PrintDirective( os ); // #omp sections

        delete sec_directive;
    }
}

void
h_omp_parallelsections( OPARI2_Directive* d_base,
                        ostream&          os )
{
    if ( opt.lang & L_FORTRAN )
    {
        h_omp_parallelsections_f( d_base, os );
    }
    else
    {
        h_omp_parallelsections_c( d_base, os );
    }
}

void
h_end_omp_parallelsections_c( OPARI2_Directive* d_base,
                              ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_FULL ) )
    {
        exit_handler_notransform( d );
    }
    else
    {
        int id = d->ExitRegion( true );


        if ( !InstrumentationDisabled( D_USER ) && ( d->active ) )
        {
            generate_barrier( id, os, d->GetFilename().c_str() );
            generate_call( "exit", "sections", id, os, NULL );
        }

        generate_call( "end", "parallel", id, os, NULL );
        generate_call_restore_task_id( "join", "parallel", id, os );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_end_omp_parallelsections_f( OPARI2_Directive* d_base,
                              ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_FULL ) )
    {
        d->PrintPlainDirective( os );
        exit_handler_notransform( d );
    }
    else
    {
        int     id       = d->ExitRegion( true );
        int     lineno   = d->GetLineno();
        string& filename = d->GetFilename();

        if ( InstrumentationDisabled( D_USER ) || !( d->active ) )
        {
            generate_directive( "end sections", lineno, filename.c_str(), os );
            //          generate_barrier( id, os, filename.c_str() );
        }
        else
        {
            generate_call( "end", "section", id, os, NULL );
            generate_directive( "end sections nowait", lineno, filename.c_str(), os );
            generate_barrier( id, os, filename.c_str() );
            generate_call( "exit", "sections", id, os, NULL );
        }
        generate_call( "end", "parallel", id, os, NULL );
        generate_directive( "end parallel", lineno, filename.c_str(), os );
        generate_call_restore_task_id( "join", "parallel", id, os );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_end_omp_parallelsections( OPARI2_Directive* d_base,
                            ostream&          os )
{
    if ( opt.lang & L_FORTRAN )
    {
        h_end_omp_parallelsections_f( d_base, os );
    }
    else
    {
        h_end_omp_parallelsections_c( d_base, os );
    }
}

void
h_omp_barrier( OPARI2_Directive* d_base,
               ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        d->PrintPlainDirective( os );
    }
    else
    {
        /** initialize region information */
        d->InitRegion();
        SaveForInit( d );

        int id = d->GetID();
        generate_call_save_task_id( "enter", "barrier", id, os, d );
        d->PrintDirective( os );
        generate_call_restore_task_id( "exit", "barrier", id, os );

        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_omp_flush( OPARI2_Directive* d_base,
             ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        d->PrintPlainDirective( os );
    }
    else
    {
        d->InitRegion();
        SaveForInit( d );

        int id = d->GetID();
        generate_call( "enter", "flush", id, os, d );
        d->PrintDirective( os );
        generate_call( "exit", "flush", id, os, NULL );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_omp_atomic( OPARI2_Directive* d_base,
              ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    SaveSingleLineDirective( d );
    /**
     * 'atomic' directive should NOT be instrumented if inside a 'workshare' region.
     */
    if ( InstrumentationDisabled( D_USER ) || in_workshare )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion();

        generate_call( "enter", "atomic", d->GetID(), os, d );
        d->PrintDirective( os );
    }
}

/**
 * Special handler for the  end of 'atomic' region in Fortran.
 */
void
extra_openmp_atomic_handler( OPARI2_Directive* d_base,
                             const int         lineno,
                             ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( d && ( InstrumentationDisabled( D_USER ) || in_workshare ) )
    {
        exit_handler_notransform( d );
        SaveSingleLineDirective( NULL );
    }
    else
    {
        if ( d )
        {
            d->SetEndLineno( lineno, lineno );
            generate_call( "exit", "atomic", d->GetID(), os, NULL );
            DirectiveStackPop();

            if ( opt.keep_src_info )
            {
                os << "#line " << ( lineno + 1 ) << " \"" << opt.infile << "\"" << "\n";
            }

            SaveSingleLineDirective( NULL );
        }
    }
}

void
h_end_omp_atomic( OPARI2_Directive* d_base,
                  ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) || in_workshare )
    {
        exit_handler_notransform( d );
        SaveSingleLineDirective( NULL );
    }
    else
    {
        int id = d->ExitRegion( false );
        generate_call( "exit", "atomic", id, os, NULL );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }

        SaveSingleLineDirective( NULL );
    }
}

void
h_omp_workshare( OPARI2_Directive* d_base,
                 ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );
    in_workshare = true;

    if ( InstrumentationDisabled( D_USER ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion();

        generate_call( "enter", "workshare", d->GetID(), os, d );
        d->PrintDirective( os );
    }
}

void
h_end_omp_workshare( OPARI2_Directive* d_base,
                     ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );
    in_workshare = false;

    if ( InstrumentationDisabled( D_USER ) )
    {
        d->PrintPlainDirective( os );
        exit_handler_notransform( d );
    }
    else
    {
        int id = d->ExitRegion( false );

        if ( d->HasClause( "nowait" ) )
        {
            d->PrintDirective( os );
            generate_call( "exit", "workshare", id, os, NULL );
        }
        else
        {
            d->AddNowait();
            d->PrintDirective( os );
            generate_barrier( id, os, d->GetFilename().c_str() );
            generate_call( "exit", "workshare", id, os, NULL );
        }

        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_omp_parallelworkshare( OPARI2_Directive* d_base,
                         ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_FULL ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion( true );

        int id = d->GetID();
        d->AddDescr( id );

        generate_fork_call( "fork", "parallel", id, os, d );
        OPARI2_DirectiveOpenmp* ws_directive = d->SplitCombined();
        print_directive_parallel( d, os ); // #omp parallel
        generate_call( "begin", "parallel", id, os, NULL );
        if ( !InstrumentationDisabled( D_USER ) )
        {
            generate_call( "enter", "workshare", id, os, d );
        }
        ws_directive->PrintDirective( os ); // #omp workshare
        in_workshare = true;

        delete ws_directive;
    }
}

/*2.0*/
void
h_end_omp_parallelworkshare( OPARI2_Directive* d_base,
                             ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );
    in_workshare = false;

    if ( InstrumentationDisabled( D_FULL ) )
    {
        d->PrintPlainDirective( os );
        exit_handler_notransform( d );
    }
    else
    {
        int     id       = d->ExitRegion( true );
        int     lineno   = d->GetLineno();
        string& filename = d->GetFilename();

        if ( InstrumentationDisabled( D_USER ) )
        {
            generate_directive( "end workshare", lineno, filename.c_str(), os );
            //generate_barrier( id, os, filename.c_str() );
        }
        else
        {
            generate_directive( "end workshare nowait", lineno, filename.c_str(), os );
            generate_barrier( id, os, filename.c_str() );
            generate_call( "exit", "workshare", id, os, NULL );
        }
        generate_call( "end", "parallel", id, os, NULL );
        generate_directive( "end parallel", lineno, filename.c_str(), os );
        generate_call_restore_task_id( "join", "parallel", id, os );
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

/*2.5*/
void
h_omp_ordered( OPARI2_Directive* d_base,
               ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion();

        int id = d->GetID();
        generate_call( "enter", "ordered", id, os, d );
        d->PrintDirective( os );
        generate_call( "begin", "ordered", id, os, NULL );

        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

/*2.5*/
void
h_end_omp_ordered( OPARI2_Directive* d_base,
                   ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        d->PrintPlainDirective( os );
        exit_handler_notransform( d );
    }
    else
    {
        int id = d->ExitRegion( false );

        generate_call( "end", "ordered", id, os, NULL );
        d->PrintDirective( os );
        generate_call( "exit", "ordered", id, os, NULL );

        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

/*3.0*/
void
h_omp_task( OPARI2_Directive* d_base,
            ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        enter_handler_notransform( d, os );
    }
    else
    {
        d->EnterRegion( true );

        const char*        inner_call, * outer_call;
        opari2_omp_option* omp_opt = OPARI2_DirectiveOpenmp::GetOpenmpOpt();


        int id = d->GetID();
        if ( !InstrumentationDisabled( D_FULL ) )
        {
            d->AddDescr( id );
        }

        outer_call = "task_create";
        inner_call = "task";

        if ( omp_opt->task_abort )
        {
            cerr << d->GetFilename() << ":" << d->GetLineno() << ":\n"
                 << "ERROR: Tasks are not allowed with this configuration." << std::endl;
            cleanup_and_exit();
        }

        if ( omp_opt->task_warn )
        {
            cerr << d->GetFilename() << ":" << d->GetLineno() << ":\n"
                 << "WARNING: Tasks may not be supported by the measurement system." << std::endl;
        }

        if ( d->HasClause( "untied" ) )
        {
            if ( omp_opt->untied_abort )
            {
                cerr << d->GetFilename() << ":" << d->GetLineno() << ":\n"
                     << "ERROR: Untied tasks are not allowed with this configuration." << std::endl;
                cleanup_and_exit();
            }
            if ( !omp_opt->untied_nowarn && !omp_opt->untied_keep )
            {
                cerr << d->GetFilename() << ":" << d->GetLineno() << ":\n"
                     << "WARNING: untied tasks may not be supported by the measurement system.\n"
                     << "         All untied tasks are now made tied.\n"
                     << "         Please consider using --omp-task-untied=abort|keep|no-warn" << std::endl;
            }

            if ( omp_opt->untied_keep )
            {
                outer_call = "untied_task_create";
                inner_call = "untied_task";
            }
        }

        if ( !omp_opt->task_remove )
        {
            if ( ( opt.lang & L_C_OR_CXX ) )
            {
                os << "{\n";
            }

            generate_if( os, d );
            generate_call_save_task_id( "begin", outer_call, id, os, d );

            stringstream adds;
            if ( opt.lang & L_FORTRAN )
            {
                if ( opt.form == F_FIX )   // Fix source form
                {
                    adds << "\n!$omp& if(pomp2_if) firstprivate(pomp2_new_task, pomp2_if)";
                    if ( d->ChangedDefault() )
                    {
                        adds << "\n!$omp& shared(/" << "cb"
                             << d->GetInodeCompiletimeID() << "/) ";
                    }
                }
                else    // Free source form
                {
                    adds << " if(pomp2_if) firstprivate(pomp2_new_task, pomp2_if)";
                    if ( d->ChangedDefault() )
                    {
                        adds << "&\n  !$omp shared(/" << "cb"
                             << d->GetInodeCompiletimeID() << "/)";
                    }
                }
            }
            else
            {
                adds << " if(pomp2_if) firstprivate(pomp2_new_task, pomp2_if)";
            }

            d->PrintDirective( os, adds.str() );
            generate_call( "begin", inner_call, id, os, NULL );
        }
        else
        {
            os << "//!!! Removed task directive due to user option \"--task=remove\"!" << std::endl;
        }
        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_end_omp_task( OPARI2_Directive* d_base,
                ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        d->PrintPlainDirective( os );
        exit_handler_notransform( d );
    }
    else
    {
        OPARI2_DirectiveOpenmp* d_top = cast2omp( DirectiveStackTop( d ) );
        int                     id    = d->ExitRegion( true );
        const char*             inner_call, * outer_call;
        opari2_omp_option*      omp_opt = OPARI2_DirectiveOpenmp::GetOpenmpOpt();

        if ( d_top->HasClause( "untied" ) && omp_opt->untied_keep )
        {
            outer_call = "untied_task_create";
            inner_call = "untied_task";
        }
        else
        {
            outer_call = "task_create";
            inner_call = "task";
        }

        if ( !omp_opt->task_remove )
        {
            generate_call( "end", inner_call, id, os, NULL );
            d->PrintDirective( os );
            generate_call_restore_task_id( "end", outer_call, id, os );
            if ( ( opt.lang & L_C_OR_CXX ) )
            {
                os << "}\n";
            }
        }

        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_omp_taskwait( OPARI2_Directive* d_base,
                ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    if ( InstrumentationDisabled( D_USER ) )
    {
        d->PrintPlainDirective( os );
    }
    else
    {
        d->InitRegion();
        SaveForInit( d );

        int id = d->GetID();
        generate_call_save_task_id( "begin", "taskwait", id, os, d );
        d->PrintDirective( os );
        generate_call_restore_task_id( "end", "taskwait", id, os );

        if ( opt.keep_src_info )
        {
            d->ResetSourceInfo( os );
        }
    }
}

void
h_omp_threadprivate( OPARI2_Directive* d_base,
                     ostream&          os )
{
    OPARI2_DirectiveOpenmp* d = cast2omp( d_base );

    d->PrintDirective( os );
}
