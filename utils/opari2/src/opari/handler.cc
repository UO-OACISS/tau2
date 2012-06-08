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
 *  @file       handler.cc
 *  @status     alpha
 *
 *  @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
 *
 *  @authors    Bernd Mohr <b.mohr@fz-juelich.de>
 *              Dirk Schmidl <schmidl@rz-rwth-aachen.de>
 *              Peter Philippen <p.philippen@fz-juelich.de>
 *
 *  @brief      This file contains all functions to instrument and print
 *              pragmas.*/

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

#include "handler.h"
#include "ompregion.h"
#include "common.h"

/*
 * global data
 */

namespace
{
class File
{
public:
    /*File( const string& n, int f, int l, Language la )
        : name( n ), first( f ), last( l ), lang( la )
       {
       }*/
    string   name;
    unsigned first, last;
    Language lang;
};

typedef map<string, phandler_t> htab;
htab               table;
vector<OMPRegion*> regions;
stack<OMPRegion*>  regStack;
OMPRegion*         atomicRegion = 0;
Language           lang         = L_NA;
bool               keepSrcInfo  = false;
const char*        infile       = "";
}

bool do_transform = true;

/*
 * local utility functions and data
 */

namespace
{
enum constructs
{
    C_NONE     = 0x0000,
    C_ATOMIC   = 0x0001,
    C_CRITICAL = 0x0002,
    C_MASTER   = 0x0004,
    C_SINGLE   = 0x0008,
    C_LOCKS    = 0x0010,
    C_FLUSH    = 0x0020,
    C_TASK     = 0x0040,
    C_ORDERED  = 0x0080,
    C_SYNC     = 0x00FF,
    C_OMP      = 0x0FFF,
    C_REGION   = 0x1000,
    C_ALL      = 0xFFFF
};

unsigned enabled = C_ALL;  // & ~C_TASK;

unsigned
string2construct( const string& str )
{
    switch ( str[ 0 ] )
    {
        case 'a':
            if ( str == "atomic" )
            {
                return C_ATOMIC;
            }
            break;
        case 'c':
            if ( str == "critical" )
            {
                return C_CRITICAL;
            }
            break;
        case 'f':
            if ( str == "flush" )
            {
                return C_FLUSH;
            }
            break;
        case 'l':
            if ( str == "locks" )
            {
                return C_LOCKS;
            }
            break;
        case 'm':
            if ( str == "master" )
            {
                return C_MASTER;
            }
            break;
        case 'o':
            if ( str == "omp" )
            {
                return C_OMP;
            }
            else if ( str == "ordered" )
            {
                return C_ORDERED;
            }
            break;
        case 'r':
            if ( str == "region" )
            {
                return C_REGION;
            }
            break;
        case 's':
            if ( str == "single" )
            {
                return C_SINGLE;
            }
            if ( str == "sync" )
            {
                return C_SYNC;
            }
            break;
        case 't':
            if ( str == "task" )
            {
                return C_TASK;
            }
            break;
    }
    return C_NONE;
}

void
generate_num_threads( ostream&   os,
                      OMPragma*  p,
                      OMPRegion* r )
{
    if ( lang & L_FORTRAN )
    {
        if ( r->has_num_threads )
        {
            os << "      pomp_num_threads = " << p->arg_num_threads << "\n";
        }
        else
        {
            os << "      pomp_num_threads = pomp2_lib_get_max_threads()\n";
        }
    }
    else
    {
        if ( r->has_num_threads )
        {
            os << "  int pomp_num_threads = " << p->arg_num_threads << ";\n";
        }
        else
        {
            os << "  int pomp_num_threads = omp_get_max_threads();\n";
        }
    }
}

void
generate_if( ostream&   os,
             OMPragma*  p,
             OMPRegion* r )
{
    if ( lang & L_FORTRAN )
    {
        if ( r->has_if )
        {
            os << "      pomp_if = ( " << p->arg_if << " )\n";
        }
        else
        {
            os << "      pomp_if = .true.\n";
        }
    }
    else
    {
        if ( r->has_if )
        {
            os << "  int pomp_if = (int)( " << p->arg_if << " );\n";
        }
        else
        {
            os << "  int pomp_if = 1;\n";
        }
    }
}

void
generate_call( const char* event,
               const char* type,
               int         id,
               ostream&    os,
               OMPRegion*  r )
{
    char   c1 = toupper( type[ 0 ] );
    string ctc_string;

    if ( lang & L_FORTRAN )
    {
        if ( strcmp( type, "task" ) == 0 || strcmp( type, "untied_task" ) == 0 )
        {
            os << "      if (pomp_if) then\n";
        }
        os << "      call POMP2_" << c1 << ( type + 1 )
           << "_" << event << "(" << region_id_prefix << id;
        if ( strstr( type, "task" ) != NULL &&
             strcmp( type, "taskwait" ) != 0 &&
             strcmp( event, "begin" ) == 0 )
        {
            os << ", pomp2_new_task";
        }
        if ( r != NULL )
        {
            if ( lang & L_F77 )
            {
                os << ",\n     &" << r->generate_ctc_string( lang ) << " ";
            }
            else
            {
                os << ", &\n     " << r->generate_ctc_string( lang ) << " ";
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
            os << "if (pomp_if)";
        }

        os << "  POMP2_" << c1 << ( type + 1 )
           << "_" << event << "( &" << region_id_prefix << id;

        if ( strstr( type, "task" ) != NULL &&
             strcmp( type, "taskwait" ) != 0 &&
             strcmp( event, "begin" ) == 0 )
        {
            os << ", pomp2_new_task";
        }

        if ( r != NULL )
        {
            os << ", " << r->generate_ctc_string( lang ) << " ";
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
generate_call_save_task_id( const char* event,
                            const char* type,
                            int         id,
                            ostream&    os,
                            OMPRegion*  r )
{
    char   c1 = toupper( type[ 0 ] );
    string ctc_string;

    if ( lang & L_FORTRAN )
    {
        if ( strcmp( type, "task_create" ) == 0 || strcmp( type, "untied_task_create" ) == 0 )
        {
            os << "      if (pomp_if) then\n";
        }
        os << "      call POMP2_" << c1 << ( type + 1 )
           << "_" << event << "(" << region_id_prefix << id;
        if ( ( strcmp( type, "task_create" ) == 0 ) || ( strcmp( type, "untied_task_create" ) == 0 )  )
        {
            os << ", pomp2_new_task";
            if ( lang & L_F77 )
            {
                os << ",\n     &pomp2_old_task";
            }
            else
            {
                os << ", &\n      pomp2_old_task";
            }
        }
        else
        {
            os << ", pomp2_old_task";
        }
        if ( r != NULL )
        {
            if ( lang & L_F77 )
            {
                if ( r->name == "task" )
                {
                    os << ", \n     &pomp_if";
                }
                os << ",\n     &" << r->generate_ctc_string( lang ) << " ";
            }
            else
            {
                if ( r->name == "task" )
                {
                    os << ", &\n      pomp_if";
                }
                os << ", &\n     " << r->generate_ctc_string( lang ) << " ";
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
            os << "if (pomp_if)";
        }
        os << "  POMP2_" << c1 << ( type + 1 )
           << "_" << event << "( &" << region_id_prefix << id;
        if ( ( strcmp( type, "task_create" ) == 0 ) || ( strcmp( type, "untied_task_create" ) == 0 )  )
        {
            os << ", &pomp2_new_task";
        }
        os << ", &pomp2_old_task";
        if ( r != NULL )
        {
            if ( r->name == "task" )
            {
                os << ", pomp_if";
            }
            os << ", " << r->generate_ctc_string( lang ) << " ";
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

    if ( lang & L_FORTRAN )
    {
        if ( strcmp( type, "task_create" ) == 0 || strcmp( type, "untied_task_create" ) == 0 )
        {
            os << "      if (pomp_if) then\n";
        }
        os << "      call POMP2_" << c1 << ( type + 1 )
           << "_" << event << "(" << region_id_prefix << id;
        os << ", pomp2_old_task" << ")\n";
        if ( strcmp( type, "task_create" ) == 0 || strcmp( type, "untied_task_create" ) == 0 )
        {
            os << "      end if\n";
        }
    }
    else
    {
        if ( strcmp( type, "task_create" ) == 0 || strcmp( type, "untied_task_create" ) == 0 )
        {
            os << "if (pomp_if)";
        }
        os << "  POMP2_" << c1 << ( type + 1 )
           << "_" << event << "( &" << region_id_prefix << id;
        os << ", pomp2_old_task ); }\n";
    }
}

/** @brief Instrument an OpenMP Fork. The pomp_num_threads variable is
 *         used to pass the number of requested threads for the
 *         parallel region to the POMP library. It is either the
 *         result of omp_get_max_threads() or of the num_threads()
 *         clause, if present.*/
void
generate_fork_call( const char* event,
                    const char* type,
                    int         id,
                    ostream&    os,
                    OMPragma*   p,
                    OMPRegion*  r )
{
    char c1 = toupper( type[ 0 ] );


    if ( lang & L_C_OR_CXX )
    {
        os << "{\n";
    }

    generate_num_threads( os, p, r );
    generate_if( os, p, r );

    if ( lang & L_FORTRAN )
    {
        os << "      call POMP2_" << c1 << ( type + 1 )
           << "_" << event << "(" << region_id_prefix << id;
        if ( lang & L_F77 )
        {
            os << ",\n     &pomp_if, pomp_num_threads, pomp2_old_task";
            if ( r != NULL )
            {
                os << ",\n     &" << r->generate_ctc_string( lang ) << " ";
            }
        }
        else
        {
            os << ",&\n      pomp_if, pomp_num_threads, pomp2_old_task";
            if ( r != NULL )
            {
                os << ", &\n      " << r->generate_ctc_string( lang ) << " ";
            }
        }
        os << ")\n";
    }
    else
    {
        os << "  POMP2_Task_handle pomp2_old_task;\n";
        os << "  POMP2_" << c1 << ( type + 1 ) << "_" << event
           << "(&" << region_id_prefix << id << ", pomp_if, pomp_num_threads, "
           << "&pomp2_old_task";
        if ( r != 0 )
        {
            os << ", " << r->generate_ctc_string( lang ) << " ";
        }
        os << ");\n";
    }
}

/** @brief Generate the OpenMP pragma. */
void
generate_pragma( const char* p,
                 ostream&    os )
{
    if ( lang & L_FORTRAN )
    {
        os << "!$omp " << p << "\n";
    }
    else
    {
        os << "#pragma omp " << p << "\n";
    }
}

void
generate_barrier( int      n,
                  ostream& os )
{
    generate_call_save_task_id( "enter", "implicit_barrier", n, os, NULL );
    generate_pragma( "barrier", os );
    generate_call_restore_task_id( "exit", "implicit_barrier", n, os );
}

/**@brief Print the OpenMP Pragma. Insert num_threads clause, if
 *        needed to pass pomp_num_therads to the OpenMP runtime. The
 *        num_threads clause, if it was presend, may not appear here
 *        again, since it would be evaluated twice and it may have
 *        side effects.*/
void
print_pragma( OMPragma* p,
              ostream&  os )
{
    if ( p->lines.size() && keepSrcInfo )
    {
        // print original source location information reset pragma
        os << "#line " << p->lineno << " \"" << infile << "\"" << "\n";
    }
    // print pragma text
    for ( unsigned i = 0; i < p->lines.size(); ++i )
    {
        os << p->lines[ i ] << "\n";
    }
}

void
print_pragma_task( OMPragma* p,
                   ostream&  os )
{
    if ( p->lines.size() && keepSrcInfo )
    {
        // print original source location information reset pragma
        os << "#line " << p->lineno << " \"" << infile << "\"" << "\n";
    }
    if ( p->name.find( "parallel" ) != string::npos && p->name.find( "end" ) == string::npos )
    {
        // print pragma text or parallel constructs
        for ( unsigned i = 0; i < p->lines.size() - 1; ++i )
        {
            os << p->lines[ i ] << "\n";
        }
        if ( lang & L_FORTRAN )
        {
            if ( lang & L_F77 )
            {
                if ( copytpd )
                {
                    os << p->lines.back() << "\n";
                    os << "!$omp& firstprivate(pomp2_old_task) private(pomp2_new_task)\n";
                    os << "!$omp& if(pomp_if) num_threads(pomp_num_threads) copyin(" << pomp_tpd << ")\n";
                    if ( p->changed_default() )
                    {
                        os << "!$omp& shared(/" << "cb" << compiletime.tv_sec << compiletime.tv_usec << "/)\n";
                    }
                }
                else
                {
                    os << p->lines.back() << "\n";
                    os << "!$omp& firstprivate(pomp2_old_task) private(pomp2_new_task)\n";
                    os << "!$omp& if(pomp_if) num_threads(pomp_num_threads) \n";
                    if ( p->changed_default() )
                    {
                        os << "!$omp& shared(/" << "cb" << compiletime.tv_sec << compiletime.tv_usec << "/)\n";
                    }
                }
            }
            else
            {
                if ( copytpd )
                {
                    os << p->lines.back() << " &\n";
                    os << "  !$omp firstprivate(pomp2_old_task) private(pomp2_new_task) &\n";
                    os << "  !$omp if(pomp_if) num_threads(pomp_num_threads) copyin(" << pomp_tpd << ")";
                    if ( p->changed_default() )
                    {
                        os << " &\n  !$omp shared(/" << "cb" << compiletime.tv_sec << compiletime.tv_usec << "/)";
                    }
                    os << "\n";
                }
                else
                {
                    os << p->lines.back() << " &\n";
                    os << "  !$omp firstprivate(pomp2_old_task) private(pomp2_new_task) &\n";
                    os << "  !$omp if(pomp_if) num_threads(pomp_num_threads)";
                    if ( p->changed_default() )
                    {
                        os << " &\n  !$omp shared(/" << "cb" << compiletime.tv_sec << compiletime.tv_usec << "/)";
                    }
                    os << "\n";
                }
            }
        }
        else
        {
            if ( copytpd )
            {
                os << p->lines.back();
                os << " firstprivate(pomp2_old_task) if(pomp_if) num_threads(pomp_num_threads) copyin(" << pomp_tpd << ")";
                if ( lang & L_FORTRAN )
                {
                    os << " private(pomp2_new_task)";
                }
                os << "\n";
            }
            else
            {
                os << p->lines.back();
                os << " firstprivate(pomp2_old_task) if(pomp_if) num_threads(pomp_num_threads)";
                if ( lang & L_FORTRAN )
                {
                    os << " private(pomp2_new_task)";
                }
                os << "\n";
            }
        }
    }
    else if ( p->name.find( "task" ) != string::npos && p->name.find( "end" ) == string::npos )
    {
        // print pragma text or parallel constructs
        for ( unsigned i = 0; i < p->lines.size() - 1; ++i )
        {
            os << p->lines[ i ] << "\n";
        }
        if ( lang & L_F77 )
        {
            os << p->lines.back() << "\n";
            os << "!$omp& if(pomp_if) firstprivate(pomp2_new_task, pomp_if)\n";
            if ( p->changed_default() )
            {
                os << "!$omp& shared(/" << "cb" << compiletime.tv_sec << compiletime.tv_usec << "/)\n ";
            }
        }
        else if ( lang & L_FORTRAN )
        {
            os << p->lines.back();
            os << " if(pomp_if) firstprivate(pomp2_new_task, pomp_if)";
            if ( p->changed_default() )
            {
                os << "&\n  !$omp shared(/" << "cb" << compiletime.tv_sec << compiletime.tv_usec << "/)";
            }
            os << "\n";
        }
        else
        {
            os << p->lines.back();
            os << " if(pomp_if) firstprivate(pomp2_new_task, pomp_if)\n";
        }
    }
    else
    {
        // print pragma text
        for ( unsigned i = 0; i < p->lines.size(); ++i )
        {
            os << p->lines[ i ] << "\n";
        }
    }
}

void
print_pragma_plain( OMPragma* p,
                    ostream&  os )
{
    for ( unsigned i = 0; i < p->lines.size(); ++i )
    {
        os << p->lines[ i ] << "\n";
    }
}

void
reset_src_info( OMPragma* p,
                ostream&  os )
{
    os << "#line " << p->lineno + p->lines.size()
       << " \"" << infile << "\"" << "\n";
}

OMPRegion*
REnter( OMPragma* p,
        bool      new_outer = false )
{
    OMPRegion* r = new OMPRegion( p->name, p->filename,
                                  p->lineno, p->lineno + p->lines.size() - 1,
                                  new_outer );
    regions.push_back( r );
    regStack.push( r );
    return r;
}

OMPRegion*
RTop( OMPragma* p )
{
    if ( regStack.empty() )
    {
        cerr << infile << ":" << p->lineno
             << "663: ERROR: unbalanced pragma/directive nesting\n";
        cleanup_and_exit();
    }
    else
    {
        return regStack.top();
    }
    return 0;
}

int
RExit( OMPragma* p,
       bool      end_outer = false )
{
    OMPRegion* r = RTop( p );

#if defined( __GNUC__ ) && ( __GNUC__ < 3 )
    if ( p->name[ 0 ] != '$' && p->name.substr( 3 ) != r->name )
    {
#else
    if ( p->name[ 0 ] != '$' && p->name.compare( 3, string::npos, r->name ) != 0 )
    {
#endif
        cerr << infile << ":" << r->begin_first_line
             << ": ERROR: missing end" << r->name << " directive for "
             << r->name << " directive\n";
        cerr << infile << ":" << p->lineno
             << ": ERROR: non-matching " << p->name
             << " directive\n";
        cleanup_and_exit();
    }
    if ( p->lines.size() )
    {
        r->end_first_line = p->lineno;
        r->end_last_line  = p->lineno + p->lines.size() - 1;
    }
    else
    {
        // C/C++ $END$ pragma
        r->end_first_line = r->end_last_line = p->lineno - p->pline;
    }
    if ( end_outer )
    {
        r->finish();
    }
    regStack.pop();

    return r->id;
}
}

/** @brief OpenMP pragma transformation functions*/
namespace
{
void
h_parallel( OMPragma* p,
            ostream&  os )
{
    OMPRegion* r = REnter( p, true );
    int        n = r->id;
    p->add_descr( n );
    r->has_num_threads = p->find_numthreads();
    r->has_if          = p->find_if();
    r->has_reduction   = p->find_reduction();
    r->has_schedule    = p->find_schedule( &( r->arg_schedule ) );
    generate_fork_call( "fork", "parallel", n, os, p, r );
    print_pragma_task( p, os );
    generate_call( "begin", "parallel", n, os, NULL );

    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_endparallel( OMPragma* p,
               ostream&  os )
{
    int n =        RExit( p, true );

    generate_barrier( n, os );
    generate_call( "end", "parallel", n, os, NULL );
    print_pragma( p, os );
    generate_call_restore_task_id( "join", "parallel", n, os );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_for( OMPragma* p,
       ostream&  os )
{
    OMPRegion* r = REnter( p );
    int        n = r->id;
    if ( !p->is_nowait() )
    {
        p->add_nowait();
        r->noWaitAdded = true;
    }
    r->has_reduction = p->find_reduction();
    r->has_schedule  = p->find_schedule( &( r->arg_schedule ) );
    r->has_collapse  = p->find_collapse();
    r->has_ordered   = p->find_ordered();

    generate_call( "enter", "for", n, os, r );
    print_pragma( p, os );
}

void
h_endfor( OMPragma* p,
          ostream&  os )
{
    OMPRegion* r = RTop( p );
    int        n = RExit( p );

    if ( r->noWaitAdded )
    {
        generate_barrier( n, os );
    }
    generate_call( "exit", "for", n, os, NULL );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_do( OMPragma* p,
      ostream&  os )
{
    OMPRegion* r = REnter( p );
    int        n = r->id;
    r->has_ordered   = p->find_ordered();
    r->has_collapse  = p->find_collapse();
    r->has_schedule  = p->find_schedule( &( r->arg_schedule ) );
    r->has_reduction = p->find_reduction();

    generate_call( "enter", "do", n, os, r );
    print_pragma( p, os );
}

void
h_enddo( OMPragma* p,
         ostream&  os )
{
    int n = RExit( p );
    if ( p->is_nowait() )
    {
        print_pragma( p, os );
    }
    else
    {
        p->add_nowait();
        print_pragma( p, os );
        generate_barrier( n, os );
    }
    generate_call( "exit", "do", n, os, NULL );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_sections_c( OMPragma* p,
              ostream&  os )
{
    OMPRegion* r = REnter( p );
    int        n = r->id;
    if ( !p->is_nowait() )
    {
        p->add_nowait();
        r->noWaitAdded = true;
    }
    r->has_reduction = p->find_reduction();

    generate_call( "enter", "sections", n, os, r );
    print_pragma( p, os );
}

void
h_section_c( OMPragma* p,
             ostream&  os )
{
    OMPRegion* r = RTop( p );
    OMPRegion* s = new OMPRegion( *r, p->name, p->filename,
                                  p->lineno, p->lineno + p->lines.size() - 1 );
    regStack.push( s );
    if ( r->num_sections )
    {
        print_pragma_plain( p, os );
    }
    else
    {
        print_pragma( p, os );
    }
    generate_call( "begin", "section", r->id, os, r );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
    r->num_sections++;
}

void
h_endsection_c( OMPragma* p,
                ostream&  os )
{
    OMPRegion* r = RTop( p );
    generate_call( "end", "section", r->id, os, NULL );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
    regStack.pop();
}

void
h_endsections_c( OMPragma* p,
                 ostream&  os )
{
    OMPRegion* r = RTop( p );
    int        n = RExit( p );
    if ( r->noWaitAdded )
    {
        generate_barrier( n, os );
    }
    generate_call( "exit", "sections", n, os, NULL );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_sections( OMPragma* p,
            ostream&  os )
{
    OMPRegion* r = REnter( p );
    int        n = r->id;
    generate_call( "enter", "sections", n, os, r );
    print_pragma( p, os );
}

void
h_section( OMPragma* p,
           ostream&  os )
{
    OMPRegion* r = RTop( p );
    if ( r->num_sections )
    {
        // close last section if necessary
        generate_call( "end", "section", r->id, os, NULL );
    }
    print_pragma( p, os );
    generate_call( "begin", "section", r->id, os, r );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
    ++( r->num_sections );
}

void
h_endsections( OMPragma* p,
               ostream&  os )
{
    int n = RExit( p );
    generate_call( "end", "section", n, os, NULL );
    if ( p->is_nowait() )
    {
        print_pragma( p, os );
    }
    else
    {
        p->add_nowait();
        print_pragma( p, os );
        generate_barrier( n, os );
    }
    generate_call( "exit", "sections", n, os, NULL );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_single_c( OMPragma* p,
            ostream&  os )
{
    OMPRegion* r = REnter( p );
    int        n = r->id;
    if ( !p->is_nowait() )
    {
        if ( !p->has_copypriv() )
        {
            p->add_nowait();
        }
        r->noWaitAdded = true;
    }
    if ( enabled & C_SINGLE )
    {
        generate_call( "enter", "single", n, os, r );
    }
    print_pragma( p, os );
    if ( enabled & C_SINGLE )
    {
        generate_call( "begin", "single", n, os, NULL );
        if ( keepSrcInfo )
        {
            reset_src_info( p, os );
        }
    }
}

void
h_endsingle_c( OMPragma* p,
               ostream&  os )
{
    OMPRegion* r = RTop( p );
    int        n = RExit( p );
    if ( enabled & C_SINGLE )
    {
        generate_call( "end", "single", n, os, NULL );
    }
    if ( r->noWaitAdded )
    {
        generate_barrier( n, os );
    }
    if ( enabled & C_SINGLE )
    {
        generate_call( "exit", "single", n, os, NULL );
        if ( keepSrcInfo )
        {
            reset_src_info( p, os );
        }
    }
}

void
h_single( OMPragma* p,
          ostream&  os )
{
    OMPRegion* r = REnter( p );
    int        n = r->id;
    if ( enabled & C_SINGLE )
    {
        generate_call( "enter", "single", n, os, r );
    }
    print_pragma( p, os );
    if ( enabled & C_SINGLE )
    {
        generate_call( "begin", "single", n, os, NULL );
        if ( keepSrcInfo )
        {
            reset_src_info( p, os );
        }
    }
}

void
h_endsingle( OMPragma* p,
             ostream&  os )
{
    int n = RExit( p );
    if ( enabled & C_SINGLE )
    {
        generate_call( "end", "single", n, os, NULL );
    }
    if ( p->is_nowait() )
    {
        print_pragma( p, os );
    }
    else
    {
        if ( !p->has_copypriv() )
        {
            p->add_nowait();
        }
        print_pragma( p, os );
        generate_barrier( n, os );
    }
    if ( enabled & C_SINGLE )
    {
        generate_call( "exit", "single", n, os, NULL );
        if ( keepSrcInfo )
        {
            reset_src_info( p, os );
        }
    }
}

void
h_master( OMPragma* p,
          ostream&  os )
{
    OMPRegion* r = REnter( p );
    int        n = r->id;
    print_pragma( p, os );
    if ( enabled & C_MASTER )
    {
        generate_call( "begin", "master", n, os, r );
        if ( keepSrcInfo )
        {
            reset_src_info( p, os );
        }
    }
}

void
h_endmaster_c( OMPragma* p,
               ostream&  os )
{
    int n = RExit( p );
    if ( enabled & C_MASTER )
    {
        generate_call( "end", "master", n, os, NULL );
        if ( keepSrcInfo )
        {
            reset_src_info( p, os );
        }
    }
}

void
h_endmaster( OMPragma* p,
             ostream&  os )
{
    int n = RExit( p );
    if ( enabled & C_MASTER )
    {
        generate_call( "end", "master", n, os, NULL );
    }
    print_pragma( p, os );
}

void
h_critical( OMPragma* p,
            ostream&  os )
{
    OMPRegion* r = REnter( p );
    int        n = r->id;
    r->sub_name = p->find_sub_name();
    if ( enabled & C_CRITICAL )
    {
        generate_call( "enter", "critical", n, os, r );
    }
    print_pragma( p, os );
    if ( enabled & C_CRITICAL )
    {
        generate_call( "begin", "critical", n, os, NULL );
    }
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_endcritical( OMPragma* p,
               ostream&  os )
{
    OMPRegion* r = RTop( p );
    int        n = RExit( p );
    if ( p->name[ 0 ] != '$' )
    {
        string cname = p->find_sub_name();
        if ( cname != r->sub_name  )
        {
            cerr << infile << ":" << r->begin_first_line
                 << ": ERROR: missing end critical(" << r->sub_name
                 << ") directive\n";
            cerr << infile << ":" << p->lineno
                 << ": ERROR: non-matching end critical(" << cname
                 << ") directive\n";
            cleanup_and_exit();
        }
    }
    if ( enabled & C_CRITICAL )
    {
        generate_call( "end", "critical", n, os, NULL );
    }
    print_pragma( p, os );
    if ( enabled & C_CRITICAL )
    {
        generate_call( "exit", "critical", n, os, NULL );
    }
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_parallelfor( OMPragma* p,
               ostream&  os )
{
    OMPRegion* r = REnter( p, true );
    int        n = r->id;
    p->add_descr( n );
    r->has_num_threads = p->find_numthreads();
    r->has_if          =          p->find_if();
    r->has_reduction   =   p->find_reduction();
    r->has_schedule    =    p->find_schedule( &( r->arg_schedule ) );
    r->has_ordered     =     p->find_ordered();
    r->has_collapse    =    p->find_collapse();

    OMPragma* forPragma = p->split_combined();
    forPragma->add_nowait();

    generate_fork_call( "fork", "parallel", n, os, p, r );
    print_pragma_task( p, os );
    generate_call( "begin", "parallel", n, os, NULL );
    generate_call( "enter", "for", n, os, r );

    print_pragma( forPragma, os );  // #omp for nowait
    delete forPragma;
}

void
h_endparallelfor( OMPragma* p,
                  ostream&  os )
{
    int n = RExit( p, true );
    generate_barrier( n, os );
    generate_call( "exit", "for", n, os, NULL );
    generate_call( "end", "parallel", n, os, NULL );
    generate_call_restore_task_id( "join", "parallel", n, os );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_paralleldo( OMPragma* p,
              ostream&  os )
{
    OMPRegion* r = REnter( p, true );
    int        n = r->id;
    p->add_descr( n );
    r->has_num_threads = p->find_numthreads();
    r->has_if          =          p->find_if();
    r->has_reduction   =   p->find_reduction();
    r->has_schedule    =    p->find_schedule( &( r->arg_schedule ) );
    r->has_ordered     =     p->find_ordered();
    r->has_collapse    =    p->find_collapse();

    generate_fork_call( "fork", "parallel", n, os, p, r );
    OMPragma* doPragma = p->split_combined();

    print_pragma_task( p, os );
    generate_call( "begin", "parallel", n, os, NULL );
    generate_call( "enter", "do", n, os, r );

    print_pragma( doPragma, os );  // #omp do
    delete doPragma;
}

void
h_endparalleldo( OMPragma* p,
                 ostream&  os )
{
    int n = RExit( p, true );
    generate_pragma( "end do nowait", os );
    generate_barrier( n, os );
    generate_call( "exit", "do", n, os, NULL );
    generate_call( "end", "parallel", n, os, NULL );
    generate_pragma( "end parallel", os );
    generate_call_restore_task_id( "join", "parallel", n, os );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_parallelsections_c( OMPragma* p,
                      ostream&  os )
{
    OMPRegion* r = REnter( p, true );
    int        n = r->id;
    p->add_descr( n );
    r->has_num_threads = p->find_numthreads();
    r->has_if          =          p->find_if();
    r->has_reduction   =   p->find_reduction();

    OMPragma* secPragma = p->split_combined();
    secPragma->add_nowait();

    generate_fork_call( "fork", "parallel", n, os, p, r );

    print_pragma_task( p, os );
    generate_call( "begin", "parallel", n, os, NULL );
    generate_call( "enter", "sections", n, os, r );

    print_pragma( secPragma, os );  // #omp sections
    delete secPragma;
}

void
h_endparallelsections_c( OMPragma* p,
                         ostream&  os )
{
    int n = RExit( p, true );
    generate_barrier( n, os );
    generate_call( "exit", "sections", n, os, NULL );
    generate_call( "end", "parallel", n, os, NULL );
    generate_call_restore_task_id( "join", "parallel", n, os );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_parallelsections( OMPragma* p,
                    ostream&  os )
{
    OMPRegion* r = REnter( p, true );
    int        n = r->id;
    p->add_descr( n );
    r->has_num_threads = p->find_numthreads();
    r->has_if          =          p->find_if();
    r->has_reduction   =   p->find_reduction();

    OMPragma* secPragma = p->split_combined();

    generate_fork_call( "fork", "parallel", n, os, p, r );

    print_pragma_task( p, os );          // #omp parallel
    generate_call( "begin", "parallel", n, os, NULL );
    generate_call( "enter", "sections", n, os, NULL );
    print_pragma( secPragma, os );  // #omp sections
    delete secPragma;
}

void
h_endparallelsections( OMPragma* p,
                       ostream&  os )
{
    int n = RExit( p, true );
    generate_call( "end", "section", n, os, NULL );
    generate_pragma( "end sections nowait", os );
    generate_barrier( n, os );
    generate_call( "exit", "sections", n, os, NULL );
    generate_call( "end", "parallel", n, os, NULL );
    generate_pragma( "end parallel", os );
    generate_call_restore_task_id( "join", "parallel", n, os );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_barrier( OMPragma* p,
           ostream&  os )
{
    OMPRegion* r = new OMPRegion( p->name, p->filename,
                                  p->lineno, p->lineno + p->lines.size() - 1 );
    int n = r->id;
    regions.push_back( r );
    generate_call_save_task_id( "enter", "barrier", n, os, r );
    print_pragma( p, os );
    generate_call_restore_task_id( "exit", "barrier", n, os );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_flush( OMPragma* p,
         ostream&  os )
{
    OMPRegion* r = new OMPRegion( p->name, p->filename,
                                  p->lineno, p->lineno + p->lines.size() - 1 );
    int n = r->id;
    regions.push_back( r );
    generate_call( "enter", "flush", n, os, r );
    print_pragma( p, os );
    generate_call( "exit", "flush", n, os, NULL );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_atomic( OMPragma* p,
          ostream&  os )
{
    OMPRegion* r = new OMPRegion( p->name, p->filename,
                                  p->lineno, p->lineno + p->lines.size() - 1 );
    int n = r->id;
    regions.push_back( r );
    if ( enabled & C_ATOMIC )
    {
        generate_call( "enter", "atomic", n, os, r );
    }
    print_pragma( p, os );
    atomicRegion = r;
}

/*2.0*/
void
h_workshare( OMPragma* p,
             ostream&  os )
{
    OMPRegion* r = REnter( p );
    int        n = r->id;

    generate_call( "enter", "workshare", n, os, r );
    print_pragma( p, os );
}

/*2.0*/
void
h_endworkshare( OMPragma* p,
                ostream&  os )
{
    int n = RExit( p );
    if ( p->is_nowait() )
    {
        print_pragma( p, os );
        generate_call( "exit", "workshare", n, os, NULL );
    }
    else
    {
        p->add_nowait();
        print_pragma( p, os );
        generate_barrier( n, os );
        generate_call( "exit", "workshare", n, os, NULL );
    }
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

/*2.0*/
void
h_parallelworkshare( OMPragma* p,
                     ostream&  os )
{
    OMPRegion* r = REnter( p, true );
    int        n = r->id;
    p->add_descr( n );
    r->has_num_threads = p->find_numthreads();
    r->has_if          =          p->find_if();
    r->has_reduction   =   p->find_reduction();

    generate_fork_call( "fork", "parallel", n, os, p, r );
    OMPragma* wsPragma = p->split_combined();
    print_pragma_task( p, os );   // #omp parallel
    generate_call( "begin", "parallel", n, os, NULL );
    generate_call( "enter", "workshare", n, os, r );

    print_pragma( wsPragma, os );  // #omp workshare
    delete wsPragma;
}

/*2.0*/
void
h_endparallelworkshare( OMPragma* p,
                        ostream&  os )
{
    int n = RExit( p, true );
    generate_pragma( "end workshare nowait", os );
    generate_barrier( n, os );
    generate_call( "exit", "workshare", n, os, NULL );
    generate_call( "end", "parallel", n, os, NULL );
    generate_pragma( "end parallel", os );
    generate_call_restore_task_id( "join", "parallel", n, os );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

/*2.5*/
void
h_ordered( OMPragma* p,
           ostream&  os )
{
    OMPRegion* r = REnter( p );
    int        n = r->id;
    if ( enabled & C_ORDERED )
    {
        generate_call( "enter", "ordered", n, os, r );
    }
    print_pragma( p, os );
    if ( enabled & C_ORDERED )
    {
        generate_call( "begin", "ordered", n, os, NULL );
        if ( keepSrcInfo )
        {
            reset_src_info( p, os );
        }
    }
}

/*2.5*/
void
h_endordered( OMPragma* p,
              ostream&  os )
{
    int n = RExit( p );
    if ( enabled & C_ORDERED )
    {
        generate_call( "end", "ordered", n, os, NULL );
    }
    print_pragma( p, os );
    if ( enabled & C_ORDERED )
    {
        generate_call( "exit", "ordered", n, os, NULL );
        if ( keepSrcInfo )
        {
            reset_src_info( p, os );
        }
    }
}

/*3.0*/
void
h_task( OMPragma* p,
        ostream&  os )
{
    OMPRegion*  r = REnter( p, true );
    int         n = r->id;
    const char* inner_call, * outer_call;

    p->add_descr( n );
    r->has_if     =     p->find_if();
    r->has_untied = p->find_untied( untied_keep );

    outer_call = "task_create";
    inner_call = "task";

    if ( task_abort )
    {
        cerr << infile << ":" << r->begin_first_line << ":\n"
             << "ERROR: Tasks are not allowed with this configuration." << std::endl;
        cleanup_and_exit();
    }

    if ( task_warn )
    {
        cerr << infile << ":" << r->begin_first_line << ":\n"
             << "WARNING: Tasks may not be supported by the measurement system." << std::endl;
    }

    if ( r->has_untied )
    {
        if ( untied_abort )
        {
            cerr << infile << ":" << r->begin_first_line << ":\n"
                 << "ERROR: Untied tasks are not allowed with this configuration." << std::endl;
            cleanup_and_exit();
        }
        if ( !untied_no_warn )
        {
            cerr << infile << ":" << r->begin_first_line << ":\n"
                 << "WARNING: untied tasks may not be supported by the measurement system.\n"
                 << "         All untied tasks are now made tied.\n"
                 << "         Please consider using --untied=abort|keep|no-warn" << std::endl;
        }

        if ( untied_keep )
        {
            outer_call = "untied_task_create";
            inner_call = "untied_task";
        }
    }

    if ( !task_remove )
    {
        if ( lang & L_C_OR_CXX )
        {
            os << "{\n";
        }

        generate_if( os, p, r );
        generate_call_save_task_id( "begin", outer_call, n, os, r );
        print_pragma_task( p, os );
        generate_call( "begin", inner_call, n, os, NULL );
    }
    else
    {
        os << "//!!! Removed task directive due to user option \"--task=remove\"!" << std::endl;
    }
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

/*3.0*/
void
h_endtask( OMPragma* p,
           ostream&  os )
{
    OMPRegion*  r = RTop( p );
    int         n = RExit( p, true );
    const char* inner_call, * outer_call;

    if ( r->has_untied && untied_keep )
    {
        outer_call = "untied_task_create";
        inner_call = "untied_task";
    }
    else
    {
        outer_call = "task_create";
        inner_call = "task";
    }

    if ( !task_remove )
    {
        generate_call( "end", inner_call, n, os, NULL );
        print_pragma( p, os );
        generate_call_restore_task_id( "end", outer_call, n, os );
        if ( lang & L_C_OR_CXX )
        {
            os << "}\n";
        }
    }

    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

/*3.0*/
void
h_taskwait( OMPragma* p,
            ostream&  os )
{
    OMPRegion* r = new OMPRegion( p->name, p->filename,
                                  p->lineno, p->lineno + p->lines.size() - 1 );
    int n = r->id;
    regions.push_back( r );
    generate_call_save_task_id( "begin", "taskwait", n, os, r );
    print_pragma( p, os );
    generate_call_restore_task_id( "end", "taskwait", n, os );
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

/*INST*/
void
h_instrument( OMPragma* p,
              ostream&  os )
{
    do_transform = true;
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

/*INST*/
void
h_noinstrument( OMPragma* p,
                ostream&  os )
{
    do_transform = false;
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

/*INST*/
void
h_inst( OMPragma* p,
        ostream&  os )
{
    char c1 = toupper( p->name.substr( 4 )[ 0 ] );
    if ( lang & L_FORTRAN )
    {
        os << "      call POMP2_" << c1 << p->name.substr( 5 ) << "()\n";
    }
    else
    {
        os << "POMP2_" << c1 << p->name.substr( 5 ) << "();\n";
    }
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

/*INST*/
void
h_instbegin( OMPragma* p,
             ostream&  os )
{
    OMPRegion* r = REnter( p );
    int        n = r->id;
    r->name     = "region";
    r->sub_name = p->find_sub_name();
    if ( lang & L_FORTRAN )
    {
        os << "      call POMP2_Begin(" << region_id_prefix << n << ")\n";
    }
    else
    {
        os << "POMP2_Begin(&" << region_id_prefix << n << ");\n";
    }
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

/*INST*/
void
h_instaltend( OMPragma* p,
              ostream&  os )
{
    OMPRegion* r     = RTop( p );
    string     cname = p->find_sub_name();
    if ( cname != r->sub_name  )
    {
        cerr << infile << ":" << r->begin_first_line
             << ": ERROR: missing inst end(" << r->sub_name
             << ") pragma/directive\n";
        cerr << infile << ":" << p->lineno
             << ": ERROR: non-matching inst end(" << cname
             << ") pragma/directive\n";
        cleanup_and_exit();
    }
    if ( lang & L_FORTRAN )
    {
        os << "      call POMP2_End(" << region_id_prefix << r->id << ")\n";
    }
    else
    {
        os << "POMP2_End(&" << region_id_prefix << r->id << ");\n";
    }
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

/*INST*/
void
h_instend( OMPragma* p,
           ostream&  os )
{
    p->name = "endregion";
    OMPRegion* r     = RTop( p );
    string     cname = p->find_sub_name();
    int        n     = RExit( p );
    if ( cname != r->sub_name  )
    {
        cerr << infile << ":" << r->begin_first_line
             << ": ERROR: missing inst end(" << r->sub_name
             << ") pragma/directive\n";
        cerr << infile << ":" << p->lineno
             << ": ERROR: non-matching inst end(" << cname
             << ") pragma/directive\n";
        cleanup_and_exit();
    }
    if ( lang & L_FORTRAN )
    {
        os << "      call POMP2_End(" << region_id_prefix << n << ")\n";
    }
    else
    {
        os << "POMP2_End(&" << region_id_prefix << n << ");\n";
    }
    if ( keepSrcInfo )
    {
        reset_src_info( p, os );
    }
}

void
h_cxx_end( OMPragma* p,
           ostream&  os )
{
    if ( atomicRegion )
    {
        if ( enabled & C_ATOMIC )
        {
            extra_handler( p->lineno - p->pline, os );
        }
    }
    else
    {
        OMPRegion* r       = RTop( p );
        phandler_t handler = find_handler( "end" + r->name );
        handler( p, os );
    }
}
}

/** @brief External interface functions*/
void
init_handler( const char* inf,
              Language    l,
              bool        g )
{
    // remember environment
    lang        = l;
    keepSrcInfo = g;
    infile      = inf;

    // init handler table
    if ( enabled & C_OMP )
    {
        if ( lang & L_FORTRAN )
        {
            table[ "do" ]           = h_do;
            table[ "enddo" ]        = h_enddo;
            table[ "workshare" ]    = h_workshare;    /*2.0*/
            table[ "endworkshare" ] = h_endworkshare; /*2.0*/
            table[ "sections" ]     = h_sections;
            table[ "section" ]      = h_section;
            table[ "endsections" ]  = h_endsections;
            table[ "single" ]       = h_single;
            table[ "endsingle" ]    = h_endsingle;
            if ( enabled & C_MASTER )
            {
                table[ "master" ]    = h_master;
                table[ "endmaster" ] = h_endmaster;
            }

            table[ "paralleldo" ]           = h_paralleldo;
            table[ "endparalleldo" ]        = h_endparalleldo;
            table[ "parallelsections" ]     = h_parallelsections;
            table[ "endparallelsections" ]  = h_endparallelsections;
            table[ "parallelworkshare" ]    = h_parallelworkshare;    /*2.0*/
            table[ "endparallelworkshare" ] = h_endparallelworkshare; /*2.0*/
        }
        else
        {
            table[ "for" ]                 = h_for;
            table[ "endfor" ]              = h_endfor;
            table[ "sections" ]            = h_sections_c;
            table[ "section" ]             = h_section_c;
            table[ "endsection" ]          = h_endsection_c;
            table[ "endsections" ]         = h_endsections_c;
            table[ "single" ]              = h_single_c;
            table[ "endsingle" ]           = h_endsingle_c;
            table[ "master" ]              = h_master;      // F version OK here
            table[ "endmaster" ]           = h_endmaster_c; // but not here
            table[ "parallelfor" ]         = h_parallelfor;
            table[ "endparallelfor" ]      = h_endparallelfor;
            table[ "parallelsections" ]    = h_parallelsections_c;
            table[ "endparallelsections" ] = h_endparallelsections_c;

            table[ "$END$" ] = h_cxx_end;
        }
        table[ "parallel" ]    = h_parallel;
        table[ "endparallel" ] = h_endparallel;
        table[ "critical" ]    = h_critical;
        table[ "endcritical" ] = h_endcritical;

        table[ "barrier" ] = h_barrier;
        if ( enabled & C_FLUSH )
        {
            table[ "flush" ] = h_flush;
        }
        table[ "atomic" ]     = h_atomic;
        table[ "ordered" ]    = h_ordered;
        table[ "endordered" ] = h_endordered;
        table[ "task" ]       = h_task;
        table[ "endtask" ]    = h_endtask;
        table[ "taskwait" ]   = h_taskwait;
    }

    if ( enabled & C_REGION )
    {
        table[ "instbegin" ]  = h_instbegin;      /*INST*/
        table[ "instaltend" ] = h_instaltend;     /*INST*/
        table[ "instend" ]    = h_instend;        /*INST*/
    }

    table[ "instinit" ]     = h_inst;         /*INST*/
    table[ "instfinalize" ] = h_inst;         /*INST*/
    table[ "inston" ]       = h_inst;         /*INST*/
    table[ "instoff" ]      = h_inst;         /*INST*/

    table[ "instrument" ]   = h_instrument;   /*INST*/
    table[ "noinstrument" ] = h_noinstrument; /*INST*/
}

void
finalize_handler( const char* incfile, char* incfileNoPath, ostream&    os )
{
    // check region stack
    if ( !regStack.empty() )
    {
        cerr << "ERROR: unbalanced pragma/directive nesting\n";
        print_regstack_top();
        cleanup_and_exit();
    }

    // generate opari include file
    ofstream incs( incfile );
    if ( !incs )
    {
        cerr << "ERROR: cannot open opari include file " << incfile << "\n";
        exit( 1 );
    }
    if ( lang & L_FORTRAN )
    {
        if ( regions.size() )
        {
            for ( unsigned i = 0; i < regions.size(); ++i )
            {
                regions[ i ]->generate_descr_f( incs, lang );
            }
        }
        OMPRegion::generate_init_handle_calls_f( os, incfileNoPath );
        OMPRegion::finalize_descrs( incs, lang );
    }
    else
    {
        if ( lang & L_C )
        {
            OMPRegion::generate_header_c( incs );
        }
        else
        {
            OMPRegion::generate_header_cxx( incs );
        }
        if ( regions.size() )
        {
            for ( unsigned i = 0; i < regions.size(); ++i )
            {
                regions[ i ]->generate_descr_c( incs );
            }
        }
        if ( lang & L_C )
        {
            OMPRegion::generate_init_handle_calls_c( incs );
        }
        else
        {
            OMPRegion::generate_init_handle_calls_cxx( incs );
        }
    }
}

phandler_t
find_handler( const string& pragma )
{
    htab::iterator it = table.find( pragma );
    if ( it != table.end() )
    {
        return it->second;
    }
    else
    {
        return print_pragma;
    }
}

void
extra_handler( int      lineno,
               ostream& os )
{
    if ( atomicRegion )
    {
        atomicRegion->end_first_line = lineno;
        atomicRegion->end_last_line  = lineno;
        generate_call( "exit", "atomic", atomicRegion->id, os, NULL );
        if ( keepSrcInfo )
        {
            os << "#line " << ( lineno + 1 ) << " \"" << infile << "\"" << "\n";
        }
        atomicRegion = 0;
    }
}

bool
set_disabled( const string& constructs )
{
    string::size_type pos  = 0;
    string::size_type last = 0;
    unsigned          c;
    string            s;

    do
    {
        pos = constructs.find_first_of( ",", last );
        if ( pos < string::npos )
        {
            s = constructs.substr( last, pos - last );
            if ( ( c = string2construct( s ) ) == C_NONE )
            {
                cerr << "ERROR: unknown value \'" << s << "\' for option --disable\n";
                return true;
            }
            else
            {
                enabled &= ~c;
            }
            last = pos + 1;
        }
    }
    while ( pos < string::npos );

    s = constructs.substr( last );
    if ( ( c = string2construct( s ) ) == C_NONE )
    {
        cerr << "ERROR: unknown value \'" << s << "\' for option --disable\n";
        return true;
    }
    else
    {
        enabled &= ~c;
    }

    return false;
}

bool
instrument_locks()
{
    return enabled & C_LOCKS;
}

bool
genLineStmts()
{
    return keepSrcInfo;
}

void
print_regstack_top()
{
    OMPRegion* rt = regStack.top();
    cerr << "       near OpenMP " << rt->name << " construct at "
         << rt->file_name << ":" << rt->begin_first_line;
    if ( rt->begin_first_line != rt->begin_last_line )
    {
        cerr << "-" << rt->begin_last_line << "\n";
    }
    else
    {
        cerr << "\n";
    }
}
