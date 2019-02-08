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
 * Copyright (c) 2009-2011, 2013, 2014, 2017,
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
/** @internal
 *
 *  @file       opari2_directive_openmp.cc
 *
 *  @brief      Methods of Openmp abstract base class,
 *              useed to process OpenMP pragmas in C/C++ and Fortran program.
 */

#include <config.h>
#include <iostream>
using std::cerr;
#include <string>
using std::string;
#include <string.h>

#include "common.h"
#include "opari2_directive_openmp.h"
#include "opari2_directive_entry_openmp.h"
#include "opari2_directive_manager.h"


#define MAKE_STR( x ) MAKE_STR_( x )
#define MAKE_STR_( x ) #x

typedef struct
{
    uint32_t     mEnum;
    const string mGroupName;
} OPARI2_OpenMPGroupStringMapEntry;

OPARI2_OpenMPGroupStringMapEntry ompGroupStringMap[] =
{
    { G_OMP_ATOMIC,   "atomic"    },
    { G_OMP_CRITICAL, "critical"  },
    { G_OMP_MASTER,   "master"    },
    { G_OMP_SINGLE,   "single"    },
    { G_OMP_LOCKS,    "locks"     },
    { G_OMP_FLUSH,    "flush"     },
    { G_OMP_TASK,     "tasks"     },
    { G_OMP_SYNC,     "sync"      },
    { G_OMP_OMP,      "omp"       },
    { G_OMP_ALL,      "all"       },
};

/**
 * @brief Handle Openmp specific command line options ONLY in the form of "--omp-xx=xxx"
 */
OPARI2_ErrorCode
OPARI2_DirectiveOpenmp::ProcessOption( string option )
{
    OPARI2_ErrorCode err_flag = OPARI2_NO_ERROR;

    if ( option != "" )
    {
        /* handle new openmp-specific options */
        if ( option == "--omp-nodecl" )
        {
            s_omp_opt.add_shared_decl = false;
        }
        else if ( option == "--omp-tpd" )
        {
            s_omp_opt.copytpd = true;
        }
        else if ( option.find( "--omp-tpd-mangling=" ) != string::npos )
        {
            size_t p1      = option.find( "=" );
            string tpd_arg = option.substr( p1 + 1 );
            if ( tpd_arg != "" )
            {
                if ( tpd_arg == "gnu"   || tpd_arg == "sun" ||
                     tpd_arg == "intel" || tpd_arg == "pgi" ||
                     tpd_arg == "cray"  )
                {
                    s_omp_opt.pomp_tpd            = "pomp_tpd_";
                    s_omp_opt.tpd_in_extern_block = false;
                }
                else if ( tpd_arg == "ibm" )
                {
                    s_omp_opt.pomp_tpd            = "pomp_tpd";
                    s_omp_opt.tpd_in_extern_block = true;
                }
                else
                {
                    cerr << "ERROR: unknown option for --omp-tpd-mangling\n";
                    err_flag = OPARI2_ERROR_WITH_MESSAGE;
                }
            }
            else
            {
                cerr << "ERROR: missing value for option --omp-tpd-mangling\n";
                err_flag = OPARI2_ERROR_WITH_MESSAGE;
            }
        }
        else if ( option.find( "--omp-task=" ) != string::npos )
        {
            size_t p1    = option.find( "=" ) + 1;
            size_t p2    = option.find( ",", p1 );
            size_t len   = p2 == string::npos ? p2 : p2 - p1;
            string token = option.substr( 11, len );
            do
            {
                if ( token == "abort" )
                {
                    s_omp_opt.task_abort = true;
                }
                else if ( token == "warn" )
                {
                    s_omp_opt.task_warn = true;
                }
                else if ( token == "remove" )
                {
                    s_omp_opt.task_remove = true;
                }
                else
                {
                    cerr << "ERROR: unknown option \"" << token << "\" for --omp-task\n";
                    err_flag = OPARI2_ERROR_WITH_MESSAGE;
                }

                p2 = option.find( ",", p1 + 1 );
                if (  p2 == string::npos )
                {
                    break;
                }
                len   = p2 - p1;
                p1    = p2 + 1;
                p2    = option.find( ",", len );
                token = option.substr( p1, p2 );
            }
            while ( true );
        }
        else if ( option.find( "--omp-task-untied=" ) != string::npos )
        {
            size_t p1    = option.find( "=" ) + 1;
            size_t p2    = option.find( ",", p1 );
            size_t len   = p2 == string::npos ? p2 : p2 - p1;
            string token = option.substr( 18, len );
            do
            {
                if ( token == "abort" )
                {
                    s_omp_opt.untied_abort = true;
                }
                else if ( token == "no-warn" )
                {
                    s_omp_opt.untied_nowarn = true;
                }
                else if ( token == "keep" )
                {
                    s_omp_opt.untied_keep = true;
                }
                else
                {
                    cerr << "ERROR: unknown option \"" << token << "\" for --omp-task-untied\n";
                    err_flag = OPARI2_ERROR_WITH_MESSAGE;
                    break;
                }

                p2 = option.find( ",", p1 + 1 );
                if (  p2 == string::npos )
                {
                    break;
                }
                len   = p2 - p1;
                p1    = p2 + 1;
                p2    = option.find( ",", len );
                token = option.substr( p1, p2 );
            }
            while ( true );
        }
        else
        {
            err_flag = OPARI2_ERROR_NO_MESSAGE;
        }
    }
    return err_flag;
}

opari2_omp_option*
OPARI2_DirectiveOpenmp::GetOpenmpOpt( void )
{
    return &s_omp_opt;
}

void
OPARI2_DirectiveOpenmp::IncrementRegionCounter( void )
{
    ++s_num_regions;
}


uint32_t
OPARI2_DirectiveOpenmp::String2Group( const string name )
{
    size_t n = sizeof( ompGroupStringMap ) / sizeof( OPARI2_OpenMPGroupStringMapEntry );

    for ( size_t i = 0; i < n; ++i )
    {
        if ( ompGroupStringMap[ i ].mGroupName.compare( name ) == 0 )
        {
            return ompGroupStringMap[ i ].mEnum;
        }
    }

    return 0;
}

/*
 * @breif Set class variable "omp_pomp_tpd".
 *
 * This static method is intended to be called by main()
 * when handling a FORTRAN source file.
 */
void
OPARI2_DirectiveOpenmp::SetOptPomptpd( string str )
{
    s_omp_opt.pomp_tpd = str;

    return;
}


/**
 * @brief Find and store the directive name.
 */
void
OPARI2_DirectiveOpenmp::FindName( void )
{
    find_name_common();

    if ( m_name == "parallel"  || m_name == "endparallel" )
    {
        string w = find_next_word();
        if ( w == "do"  || w == "sections" ||
             w == "for" || w == "workshare" /*2.0*/ )
        {
            m_name += w;
        }
    }
    else if ( m_name == "end" )
    {
        string w = find_next_word();
        m_name += w;
        if ( w == "parallel" )
        {
            w = find_next_word();
            if ( w == "do"  || w == "sections" ||
                 w == "for" || w == "workshare" /*2.0*/ )
            {
                m_name += w;
            }
        }
    }

    if ( m_name == "critical" ||
         m_name == "endcritical" )
    {
        string cname = find_next_word();
        if ( cname == "(" )
        {
            m_user_name = find_next_word();
        }
    }

    identify_clauses();
}

void
OPARI2_DirectiveOpenmp::identify_clauses( void )
{
    for ( vector<string>::iterator it = s_directive_clauses[ m_name ].begin(); it != s_directive_clauses[ m_name ].end(); ++it )
    {
        unsigned          line = 0;
        string::size_type pos  = 0;

        if ( find_word( *it, line, pos ) )
        {
            bool remove = ( *it == "if"          ||
                            *it == "num_threads" ||
                            ( *it == "untied" && !s_omp_opt.untied_keep ) );
            m_clauses[ *it ] = find_arguments( line, pos, remove, *it );
        }
    }
}

string
OPARI2_DirectiveOpenmp::generate_ctc_string( OPARI2_Format_t form )
{
    stringstream s;

    s.clear();

    if ( m_name == "critical" )
    {
        if ( !m_user_name.empty() )
        {
            s << "criticalName=" << m_user_name << "*";
        }
    }
    else if ( m_name == "sections" )
    {
        s << "numSections=" << m_num_sections << "*";
    }

    for ( OPARI2_StrStr_map_t::iterator it = m_clauses.begin(); it != m_clauses.end(); ++it )
    {
        /** Clauses not to appear on CTC-string */
        if ( ( *it ).first != "private"      &&
             ( *it ).first != "lastprivate"  &&
             ( *it ).first != "firstprivate" &&
             ( *it ).first != "copyin"       &&
             ( ( *it ).first != "default" || ChangedDefault() ) )
        {
            string name = ( *it ).first;
            name[ 0 ] = toupper( name[ 0 ] );
            s << "has" << name << "=";
            /** Clauses for which the argument is put on the
                CTC-string */
            if ( ( *it ).first == "schedule"    ||
                 ( *it ).first == "default" )
            {
                /*replace * with @ in the CTC String to distinguish it from the CTC String delimited '*' */
                while ( ( *it ).second.find( '*' ) != string::npos )
                {
                    ( *it ).second.replace( ( *it ).second.find( '*' ), 1, "@" );
                }
                s << ( *it ).second << "*";
            }
            else
            {
                s << "1" << "*";
            }
        }
    }

    return generate_ctc_string_common( form, s.str() );
}


/**
 * @brief Is the default data sharing changed by default(none) or default(private) clause?
 */
bool
OPARI2_DirectiveOpenmp::ChangedDefault( void )
{
    if ( m_clauses.find( "default" ) == m_clauses.end() )
    {
        return false;
    }
    else
    {
        return m_clauses[ "default" ] == "none" ||
               m_clauses[ "default" ] == "private";
    }
}


void
OPARI2_DirectiveOpenmp::FindUserName( void )
{
    string cname = find_next_word();
    if ( cname == "(" )
    {
        m_user_name = find_next_word();
    }
}


string&
OPARI2_DirectiveOpenmp::GetUserName( void )
{
    return m_user_name;
}


void
OPARI2_DirectiveOpenmp::SetNumSections( int n )
{
    m_num_sections = n;
}


int
OPARI2_DirectiveOpenmp::GetNumSections( void )
{
    return m_num_sections;
}


bool
OPARI2_DirectiveOpenmp::IsNowaitAdded( void )
{
    return m_nowait_added;
}


void
OPARI2_DirectiveOpenmp::GenerateDescr( ostream& os )
{
    if ( s_lang & L_FORTRAN )
    {
        s_init_handle_calls << "         call " << s_paradigm_prefix
                            << "_Assign_handle( "
                            << region_id_prefix << m_id << ", ";
        if ( s_format == F_FIX )
        {
            s_init_handle_calls << "\n     &   ";
        }
        else
        {
            s_init_handle_calls << "&\n         ";
        }

        s_init_handle_calls << m_ctc_string_variable << " )\n";
    }
    else if ( s_lang & L_C_OR_CXX )
    {
        if ( !m_descrs.empty() )
        {
            stringstream ids;
            ids << m_id;

            os << "#define POMP2_DLIST_" << string( 5 - ids.str().length(), '0' )
               << ids.str() << " shared(";

            for ( set<int>::const_iterator it = m_descrs.begin();
                  it != m_descrs.end(); ++it )
            {
                if ( it != m_descrs.begin() )
                {
                    os << ",";
                }
                os << region_id_prefix << *it;
            }
            os << ")\n";
        }

        s_init_handle_calls << "    " << s_paradigm_prefix << "_Assign_handle( "
                            << "&" << region_id_prefix << m_id  << ", "
                            << m_ctc_string_variable << " );\n";
    }

    OPARI2_Directive::generate_descr_common( os );
}


void
OPARI2_DirectiveOpenmp::GenerateHeader( ostream& os )
{
    if ( s_lang & L_C )
    {
        if ( !s_preprocessed_file )
        {
            os << "#include <opari2/pomp2_lib.h>\n\n";
        }

        if ( s_omp_opt.copytpd )
        {
            if ( !s_preprocessed_file )
            {
                os << "#include <stdint.h>\n";
            }
            os << "extern int64_t " << MAKE_STR( FORTRAN_ALIGNED ) " " << s_omp_opt.pomp_tpd << ";\n";
            os << "#pragma omp threadprivate(" << s_omp_opt.pomp_tpd << ")\n";
        }
    }
    else if ( s_lang & L_CXX )
    {
        if ( !s_preprocessed_file )
        {
            os << "#include <opari2/pomp2_lib.h>\n\n";
        }
        if ( s_omp_opt.copytpd )
        {
            if ( !s_preprocessed_file )
            {
                os << "#include <stdint.h>\n";
            }
            os << "extern \"C\" \n{\n";
            os << "extern int64_t " << MAKE_STR( FORTRAN_ALIGNED ) " " << s_omp_opt.pomp_tpd << ";\n";
            if ( !s_omp_opt.tpd_in_extern_block )
            {
                os << "}\n";
            }
            os << "#pragma omp threadprivate(" << s_omp_opt.pomp_tpd << ")\n";
            if ( s_omp_opt.tpd_in_extern_block )
            {
                os << "}\n";
            }
        }
    }
}


/**
 * @brief Generate a function to allow initialization of all region handles for Fortran.
 *
 * These functions need to be called from POMP2_Init_regions.
 */
void
OPARI2_DirectiveOpenmp::GenerateInitHandleCalls( ostream&     os,
                                                 const string incfile )
{
    OPARI2_Directive::GenerateInitHandleCalls( os, incfile, s_paradigm_prefix,
                                               s_init_handle_calls, s_num_regions );
    return;
}


/** @brief add a nowait to a pragma */
void
OPARI2_DirectiveOpenmp::AddNowait( void )
{
    if ( s_lang & L_FORTRAN )
    {
        int               lastline = m_lines.size() - 1;
        string::size_type s        = m_lines[ lastline ].find( m_directive_prefix[ 0 ] ) + m_directive_prefix[ 0 ].length();
        // insert on last line on last position before comment
        string::size_type c = m_lines[ lastline ].find( '!', s );
        if ( c == string::npos )
        {
            m_lines[ lastline ].append( " nowait" );
        }
        else
        {
            m_lines[ lastline ].insert( c, " nowait" );
        }

        m_nowait_added = true;
    }
    else if ( s_lang & L_C_OR_CXX )
    {
        int lastline = m_lines.size() - 1;
        m_lines[ lastline ].append( " nowait" );

        m_nowait_added = true;
    }
}


/** @brief add region descriptors to shared variable  list*/
void
OPARI2_DirectiveOpenmp::AddDescr( int id )
{
    if ( s_lang & L_C_OR_CXX )
    {
        std::ostringstream os;
        if ( s_omp_opt.add_shared_decl )
        {
            std::stringstream ids;
            ids << id;

            os << " POMP2_DLIST_" << string( 5 - ids.str().length(), '0' ) << ids.str();
        }
        else
        {
            // not 100% right but best we can do if compiler doesn't allow
            // macro replacement on pragma statements
            os << " shared(" << region_id_prefix << id << ")";
        }
        int lastline = m_lines.size() - 1;
        m_lines[ lastline ].append( os.str() );
    }
}


/**
 * @brief Split combined parallel and worksharing constructs in two
 *         seperate pragmas to allow the insertion of POMP function
 *         calles between the parallel and the worksharing construct.
 *         clauses need to be matched to the corresponding pragma
 */
OPARI2_DirectiveOpenmp*
OPARI2_DirectiveOpenmp::SplitCombined( void )
{
    return SplitCombinedT<OPARI2_DirectiveOpenmp>( s_outer_inner, s_inner_clauses );
}


void
OPARI2_DirectiveOpenmp::FinalizeDescrs( ostream& os )
{
    if ( s_lang & L_FORTRAN )
    {
        if ( s_lang & L_F77 )
        {
            os << "\n      integer*4 pomp2_lib_get_max_threads";
            os << "\n      logical pomp2_test_lock";
            os << "\n      integer*4 pomp2_test_nest_lock\n";
        }
        else if ( s_lang & L_F90 )
        {
            os << "\n      integer ( kind=4 ) :: pomp2_lib_get_max_threads";
            os << "\n      logical :: pomp2_test_lock";
            os << "\n      integer ( kind=4 ) :: pomp2_test_nest_lock\n";
        }

        if ( !s_common_block.empty() )
        {
            if ( s_omp_opt.copytpd )
            {
                if ( s_lang & L_F77 )
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

            if ( s_lang & L_F77 )
            {
                os << "      integer*8 pomp2_old_task, pomp2_new_task \n";
                os << "      logical pomp2_if \n";
                os << "      integer*4 pomp2_num_threads \n";
            }
            else
            {
                os << "      integer ( kind=8 ) :: pomp2_old_task, pomp2_new_task \n";
                os << "      logical :: pomp2_if \n";
                os << "      integer ( kind=4 ) :: pomp2_num_threads \n";
            }
        }
    }
}


/** Writes Directive for ending a loop */
OPARI2_Directive*
OPARI2_DirectiveOpenmp::EndLoopDirective( const int lineno )
{
    if ( s_lang & L_FORTRAN )
    {
        FindName();

        string pragma = "";
        for ( string::size_type c = 0; c < m_indent; c++ )
        {
            pragma += " ";
        }

        pragma += m_directive_prefix[ 0 ];

        if ( m_name == "do" )
        {
            pragma += " end do ";
        }
        else if ( m_name == "paralleldo" )
        {
            pragma += " end parallel do ";
        }

        vector<string> lines;
        lines.push_back( pragma );

        return new OPARI2_DirectiveOpenmp( m_filename, lineno,
                                           lines, m_directive_prefix );
    }

    return NULL;
}

/** Returns true for enddo and endparalleldo directives */
bool
OPARI2_DirectiveOpenmp::EndsLoopDirective( void )
{
    if ( s_lang & L_FORTRAN )
    {
        if ( m_name.empty() )
        {
            FindName();
        }

        if ( m_name == "enddo" || m_name == "endparalleldo" )
        {
            return true;
        }
    }

    return false;
}


/* Initialize class static variables */
#ifndef FORTRAN_MANGLED
#define FORTRAN_MANGLED( str ) str
#endif

#define OPARI2_STR_( str ) #str
#define OPARI2_STR( str ) OPARI2_STR_( str )

/**
 *  @brief Creates pairs of directive keywords that can be combined
 *         and are split by OPARI2.
 *
 *  This function is called during initialiazation of static
 *  variables, so the language is not yet set in s_lang.
 */
static OPARI2_StrVStr_map_t
make_directive_clauses( void )
{
    OPARI2_StrVStr_map_t dc;

    vector<string> clauses;

    clauses.push_back( "if" );
    clauses.push_back( "num_threads" );
    clauses.push_back( "default" );
    clauses.push_back( "private" );
    clauses.push_back( "firstprivate" );
    clauses.push_back( "shared" );
    clauses.push_back( "copyin" );
    clauses.push_back( "reduction" );
    clauses.push_back( "proc_bind" );
    dc[ "parallel" ] = clauses;

    clauses.clear();
    clauses.push_back( "private" );
    clauses.push_back( "firstprivate" );
    clauses.push_back( "lastprivate" );
    clauses.push_back( "reduction" );
    clauses.push_back( "schedule" );
    clauses.push_back( "collapse" );
    clauses.push_back( "ordered" );
    clauses.push_back( "nowait" );
    dc[ "for" ] = clauses;

    clauses.clear();
    clauses.push_back( "private" );
    clauses.push_back( "firstprivate" );
    clauses.push_back( "astprivate" );
    clauses.push_back( "reduction" );
    clauses.push_back( "schedule" );
    clauses.push_back( "collapse" );
    clauses.push_back( "ordered" );
    dc[ "do" ] = clauses;

    clauses.clear();
    clauses.push_back( "private" );
    clauses.push_back( "firstprivate" );
    clauses.push_back( "lastprivate" );
    clauses.push_back( "reduction" );
    clauses.push_back( "nowait" );
    dc[ "sections" ] = clauses;

    clauses.clear();
    clauses.push_back( "private" );
    clauses.push_back( "firstprivate" );
    clauses.push_back( "copyprivate" );
    clauses.push_back( "nowait" );
    dc[ "single" ] = clauses;

    clauses.clear();
    clauses.push_back( "nowait" );
    clauses.push_back( "copyprivate" );
    dc[ "end single" ] = clauses;
    dc[ "endsingle" ]  = clauses;

    clauses.clear();
    clauses.push_back( "nowait" );
    dc[ "end do" ]        = clauses;
    dc[ "enddo" ]         = clauses;
    dc[ "end sections" ]  = clauses;
    dc[ "endsections" ]   = clauses;
    dc[ "end workshare" ] = clauses;
    dc[ "endworkshare" ]  = clauses;

    /** Commented out are the ones that are new with the OpenMP 4.0
        specification */
    clauses.clear();
    clauses.push_back( "if" );
    //clauses.push_back( "final" );
    clauses.push_back( "untied" );
    clauses.push_back( "default" );
    //clauses.push_back( "mergeable" );
    clauses.push_back( "private" );
    clauses.push_back( "firstprivate" );
    clauses.push_back( "shared" );
    //clauses.push_back( "depend" );
    dc[ "task" ] = clauses;

    dc[ "parallelfor" ] = dc[ "parallel" ];
    dc[ "parallelfor" ].insert( dc[ "parallelfor" ].end(),
                                dc[ "for" ].begin(), dc[ "for" ].end() );

    dc[ "paralleldo" ] = dc[ "parallel" ];
    dc[ "paralleldo" ].insert( dc[ "paralleldo" ].end(),
                               dc[ "do" ].begin(), dc[ "do" ].end() );

    dc[ "parallelsections" ] = dc[ "parallel" ];
    dc[ "parallelsections" ].insert( dc[ "parallelsections" ].end(),
                                     dc[ "sections" ].begin(), dc[ "sections" ].end() );

    dc[ "parallelworkshare" ] = dc[ "parallel" ];
    dc[ "parallelworkshare" ].insert( dc[ "parallelworkshare" ].end(),
                                      dc[ "workshare" ].begin(), dc[ "workshare" ].end() );

    /** Doing the work anyway and usintg the OpenMP 4.0 specification, I
        went ahead and added these clauses too */
    // clauses.clear();
    // clauses.push_back( "safelen" );
    // clauses.push_back( "linear" );
    // clauses.push_back( "aligned" );
    // clauses.push_back( "private" );
    // clauses.push_back( "lastprivate" );
    // clauses.push_back( "reduction" );
    // clauses.push_back( "collapse" );
    // dc[ "simd" ] = clauses;

    // clauses.clear();
    // clauses.push_back( "simdlen" );
    // clauses.push_back( "linear" );
    // clauses.push_back( "aligned" );
    // clauses.push_back( "uniform" );
    // clauses.push_back( "inbranch" );
    // clauses.push_back( "notinbranch" );
    // dc[ "declare simd" ] = clauses;

    // clauses.clear();
    // clauses.push_back( "device" );
    // clauses.push_back( "map" );
    // clauses.push_back( "if" );
    // dc[ "target data" ] = clauses;

    // clauses.clear();
    // clauses.push_back( "device" );
    // clauses.push_back( "map" );
    // clauses.push_back( "if" );
    // dc[ "target" ] = clauses;

    // clauses.clear();
    // clauses.push_back( "to" );
    // clauses.push_back( "from" );
    // clauses.push_back( "device" );
    // clauses.push_back( "if" );
    // dc[ "target update" ] = clauses;

    // clauses.clear();
    // clauses.push_back( "num_teams" );
    // clauses.push_back( "thread_limit" );
    // clauses.push_back( "default" );
    // clauses.push_back( "private" );
    // clauses.push_back( "firstprivate" );
    // clauses.push_back( "shared" );
    // clauses.push_back( "reduction" );
    // dc[ "teams" ] = clauses;

    // clauses.clear();
    // clauses.push_back( "private" );
    // clauses.push_back( "firstprivate" );
    // clauses.push_back( "collapse" );
    // clauses.push_back( "dist_schedule" );
    // dc[ "distribute" ] = clauses;

    // clauses.clear();
    // clauses.push_back( "parallel" );
    // clauses.push_back( "sections" );
    // clauses.push_back( "for" );
    // clauses.push_back( "do" );
    // clauses.push_back( "taskgroup" );
    // clauses.push_back( "if  " );
    // dc[ "cancel" ] = clauses;

    // clauses.clear();
    // clauses.push_back( "parallel" );
    // clauses.push_back( "sections" );
    // clauses.push_back( "for" );
    // clauses.push_back( "do" );
    // clauses.push_back( "taskgroup" );
    // dc[ "cancellation point" ] = clauses;

    return dc;
}

OPARI2_StrVStr_map_t OPARI2_DirectiveOpenmp::s_directive_clauses = make_directive_clauses();


/**
 *  @brief Creates pairs of directive keywords that can be combined
 *         and are split by OPARI2.
 *
 *  This function is called during initialiazation of static
 *  variables, so the language is not yet set in s_lang.
 */
static OPARI2_StrStr_pairs_t
make_outer_inner( void )
{
    OPARI2_StrStr_pairs_t oi;

    oi.push_back( make_pair( string( "parallel" ), string( "sections" ) ) );

    /* These are Fortran specific */
    oi.push_back( make_pair( string( "parallel" ), string( "do" ) ) );
    oi.push_back( make_pair( string( "parallel" ), string( "workshare" ) ) );

    /* This one is C/C++ specific */
    oi.push_back( make_pair( string( "parallel" ), string( "for" ) ) );

    return oi;
}

OPARI2_StrStr_pairs_t OPARI2_DirectiveOpenmp:: s_outer_inner = make_outer_inner();


/**
 * @brief Specifies the clauses that belong to the "inner" directive
 *        of a combined directive.
 */
static OPARI2_StrBool_pairs_t
make_inner_clauses( void )
{
    OPARI2_StrBool_pairs_t ic;

    ic.push_back( make_pair( string( "ordered" ), false ) );
    /* Comment out due to ticket #939 in Score-P (silc) The same
       treatment does not work for lastprivate, although this does not
       currently work. (Philippen)
       TODO: fix */
    //    ic.push_back( make_pair( string( "firstprivate" ), true ) );
    ic.push_back( make_pair( string( "lastprivate" ), true ) );
    ic.push_back( make_pair( string( "schedule" ), true ) );
    ic.push_back( make_pair( string( "collapse" ), true ) );

    return ic;
}

OPARI2_StrBool_pairs_t OPARI2_DirectiveOpenmp::s_inner_clauses = make_inner_clauses();

#define POMP_TPD_MANGLED FORTRAN_MANGLED( pomp_tpd )
opari2_omp_option OPARI2_DirectiveOpenmp::s_omp_opt = { true,  false,  false,
                                                        false, false,  false,
                                                        false, false,  false,
                                                        OPARI2_STR( POMP_TPD_MANGLED ) };

stringstream OPARI2_DirectiveOpenmp:: s_init_handle_calls;
int          OPARI2_DirectiveOpenmp::          s_num_regions = 0;
const string OPARI2_DirectiveOpenmp:: s_paradigm_prefix      = "POMP2";
