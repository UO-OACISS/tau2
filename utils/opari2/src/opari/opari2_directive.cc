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
/** @internal
 *
 *  @file		opari2_directive.cc
 *
 *  @brief		Methods of abstract base classs 'OPARI2_Directive'.
 *
 */

#include <config.h>
#include <iostream>
#include <string>
#include <assert.h>
using std::cerr;
#include <algorithm>
using std::transform;
using std::remove_if;

#include "common.h"
#include "opari2.h"
#include "opari2_directive.h"
#include "opari2_directive_manager.h"


OPARI2_Directive::OPARI2_Directive( const string&   fname,
                                    const int       ln,
                                    vector<string>& lines,
                                    vector<string>& directive_prefix )
    : m_filename( fname ), m_begin_first_line( ln ),
    m_lines( lines ), m_directive_prefix( directive_prefix )
{
    m_pline = 0;
    m_ppos  = 0;

    DelInlineComments();

    for ( vector<string>::iterator dp = directive_prefix.begin(); dp != directive_prefix.end(); ++dp )
    {
        string token = find_next_word();

        if ( token != *dp )
        {
            std::cerr << "ERROR: Directive token mismatch!\n";
            cleanup_and_exit();
        }
    }

    m_orig_lines = lines;
    m_name.clear();
    m_needs_end_loop_directive = false;
}


OPARI2_ParadigmType_t
OPARI2_Directive::GetParadigmType( void )
{
    return m_type;
}


string&
OPARI2_Directive::GetName( void )
{
    if ( m_name.empty() )
    {
        FindName();
    }
    if ( m_name == "" )
    {
        m_name = "$END$";
    }
    return m_name;
}


/** @brief Returns whether a clause is present */
bool
OPARI2_Directive::HasClause( const string& clause )
{
    return m_clauses.find( clause ) != m_clauses.end();
}


/** @brief Returns the argument of a clause */
string
OPARI2_Directive::GetClauseArg( const string& clause )
{
    if ( m_clauses.find( clause ) != m_clauses.end() )
    {
        return m_clauses[ clause ];
    }
    else
    {
        return "";
    }
}


void
OPARI2_Directive::DelInlineComments( void )
{
    if ( s_lang & L_FORTRAN )
    {
        for ( vector<string>::iterator l = m_lines.begin(); l != m_lines.end(); ++l )
        {
            // find first !
            int c = ( *l ).find( "!" );

            for ( unsigned i = c + 1; i < ( *l ).size(); ++i )
            {
                // zero out string constants and free form comments
                if ( ( *l )[ i ] == '!' )
                {
                    /* -- zero out partial line F90 comments -- */
                    for (; i < ( *l ).size(); ++i )
                    {
                        ( *l )[ i ] = ' ';
                    }
                    break;
                }
            }
        }
    }
}


string&
OPARI2_Directive::GetFilename( void )
{
    return m_filename;
}


int&
OPARI2_Directive::GetLineno( void )
{
    return m_begin_first_line;
}

void
OPARI2_Directive::SetEndLineno( const int endline_begin,
                                const int endline_end )
{
    m_end_first_line = endline_begin;
    m_end_last_line  = endline_end;
}


void
OPARI2_Directive::ResetSourceInfo( ostream& os )
{
    if ( m_name == "$END$" )
    {
        os << "#line " << m_begin_first_line + 1
           << " \"" << m_filename << "\"" << "\n";
    }
    else
    {
        os << "#line " << m_begin_first_line + m_lines.size()
           << " \"" << m_filename << "\"" << "\n";
    }
}


void
OPARI2_Directive::EnterRegion( bool new_outer,
                               bool save_on_vec )
{
    /** initialize region information */
    InitRegion( new_outer );

    if ( save_on_vec )
    {
        SaveForInit( this );
    }

    DirectiveStackPush( this );
    this->IncrementRegionCounter();
}



void
OPARI2_Directive::InitRegion( bool outer )
{
    m_id = ++s_num_all_regions;

    /** by default, region name is the same as directive name */
    m_begin_last_line = m_begin_first_line + m_lines.size() - 1;
    m_end_first_line  = m_begin_first_line;
    m_end_last_line   = m_begin_first_line + m_lines.size() - 1;
    m_outer_reg       = outer;

    // to keep track of nested directives
    m_enclosing = s_outer;

    if ( m_outer_reg )
    {
        s_outer = this;
    }
    if ( s_outer )
    {
        s_outer->InsertDescr( m_id );
    }
    DirectiveStackInsertDescr( m_id );

    stringstream stream;
    stream << string_id_prefix << m_id;
    m_ctc_string_variable = stream.str();
}


void
OPARI2_Directive::InitRegion( OPARI2_Directive* parent,
                              bool              outer )
{
    --s_num_all_regions;
    InitRegion( outer );
    m_id = parent->GetID();
}


int
OPARI2_Directive::ExitRegion( bool end_outer )
{
    OPARI2_Directive* d_top = DirectiveStackTop( this );

    assert( d_top != NULL );

    string& name_top = d_top->GetName();

    /**
     * For FORTRAN, some directive block ends with "end directive_name"
     */


#if defined( __GNUC__ ) && ( __GNUC__ < 3 )
    if ( m_name[ 0 ] != '$' && m_name.substr( 3 ) != name_top )
    {
#else
    if ( m_name[ 0 ] != '$' && m_name.compare( 3, string::npos, name_top ) != 0 )
    {
#endif
        cerr << m_filename << ":" << d_top->GetLineno()
             << ": ERROR: missing end" << name_top << " directive for "
             << name_top << " directive\n";
        cerr << m_filename << ":" << m_begin_first_line
             << ": ERROR: non-matching " << m_name
             << " directive\n";
        cleanup_and_exit();
    }

    /** Set end line information of the corresponding region */
    if ( m_lines.size() )
    {
        d_top->SetEndLineno( m_begin_first_line, m_begin_first_line + m_lines.size() - 1 );
    }
    else
    {
        // C/C++ $END$ pragma
        d_top->SetEndLineno( m_begin_first_line, m_begin_first_line );
    }
    if ( end_outer )
    {
        d_top->FinishRegion();
    }

    int region_id = d_top->GetID();
    DirectiveStackPop();

    return region_id;
}


void
OPARI2_Directive::InsertDescr( int descr )
{
    m_descrs.insert( descr );
}

bool
OPARI2_Directive::DescrsEmpty()
{
    return m_descrs.empty();
}

void
OPARI2_Directive::FinishRegion( void )
{
    if ( m_outer_reg )
    {
        s_outer = m_enclosing;
    }
}


bool
OPARI2_Directive::NeedsEndLoopDirective( void )
{
    return m_needs_end_loop_directive;
}


void
OPARI2_Directive::NeedsEndLoopDirective( bool val )
{
    m_needs_end_loop_directive = val;
}


void
OPARI2_Directive::PrintDirective( ostream&      os,
                                  const string& adds )
{
    if ( m_lines.size() && s_keep_src_info && !InstrumentationDisabled( D_USER ) )
    {
        // print original source location information reset pragma
        os << "#line " << m_begin_first_line << " \"" << m_filename << "\"" << "\n";
    }

    PrintPlainDirective( os, adds );
}


void
OPARI2_Directive::PrintPlainDirective( ostream&      os,
                                       const string& adds )
{
    // print pragma text
    if ( m_lines.size() )
    {
        for ( unsigned i = 0; i < m_lines.size() - 1; ++i )
        {
            os << m_lines[ i ] << "\n";
        }

        os << m_lines.back() << adds << "\n";
    }
}

/* *INDENT-OFF* */
/**
 * The generated code might look like this (OpenMP example):
 *
 * @code

   #ifdef __cplusplus
   extern "C"
   #endif
   void POMP2_Init_reg_000()
   {
     POMP2_Assign_handle( &opari2_region_1, opari2_ctc_1 );
     ...
     ...
   }

 * @endcode
 *
 * or in Fortran:
 *
 * @code

   subroutine POMP2_Init_reg_000()
   include 'test7.f90.opari.inc'
   call POMP2_Assign_handle( opari2_region_1, &
   opari2_ctc_1 )
   ...
   ...
   end

 * @endcode
 *
 * This generated function needs to be called from
 * POMP2_Init_regions. There will be one of these for each compile
 * unit.
 */
/* *INDENT-ON* */
void
OPARI2_Directive::GenerateInitHandleCalls( ostream&            os,
                                           const string        incfile,
                                           const string        paradigm_prefix,
                                           const stringstream& init_handle_calls,
                                           const int           num_regions )
{
    if ( num_regions > 0 )
    {
        if ( s_lang & L_FORTRAN )
        {
            if ( init_handle_calls.rdbuf()->in_avail() != 0 )
            {
                //add a Function to initialize the handles at the end of the file
                os << "\n      subroutine " << paradigm_prefix << "_Init_reg_"
                   << s_inode_compiletime_id
                   << "_" << num_regions << "()\n"
                   << "         include \'" << incfile << "\'\n"
                   << init_handle_calls.str()
                   << "      end\n";
            }
        }
        else if ( s_lang & L_C_OR_CXX )
        {
            if ( s_lang & L_C )
            {
                os << "\n#ifdef __cplusplus \n extern \"C\" \n#endif";
            }
            else if ( s_lang & L_CXX )
            {
                os << "extern \"C\" \n{";
            }

            os << "\nvoid " << paradigm_prefix << "_Init_reg_"
               << s_inode_compiletime_id
               << "_" << num_regions << "()\n{\n"
               << init_handle_calls.str();

            if ( s_lang & L_C )
            {
                os << "}\n";
            }
            else if ( s_lang & L_CXX )
            {
                os << "}\n}\n";
            }
        }
    }
}


void
OPARI2_Directive::FinalizeFortranDescrs( ostream& os )
{
    if ( !s_common_block.empty() )
    {
        vector<int>::iterator it = s_common_block.begin();

        os << "      common /" << "cb"
           << s_inode_compiletime_id
           << "/ " << region_id_prefix << *it++;

        for (; it < s_common_block.end(); it++ )
        {
            if ( s_format == F_FIX )
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


void
OPARI2_Directive::find_name_common( void )
{
    if ( m_lines.empty() )
    {
        // automatically generated END pragma for C/C++
        m_name = "$END$";
        return;
    }

    if ( !m_name.empty() )
    {
        return;
    }

    m_name = find_next_word();
}



string
OPARI2_Directive::generate_ctc_string_common( OPARI2_Format_t form,
                                              string          specific_part )
{
    stringstream stream1, stream2;

    stream1 << "*regionType=" << m_name << "*"
            << "sscl="  << m_filename << ":"
            << m_begin_first_line << ":"
            << m_begin_last_line << "*"
            << "escl="  << m_filename << ":"
            << m_end_first_line << ":"
            << m_end_last_line << "*"
            << specific_part << "*";

    stream2 << "\"" << stream1.str().length() + 2 << stream1.str() << "\"";
    m_ctc_string_len = stream2.str().length();

    string ctc_string = stream2.str();
    if ( form == F_FIX )
    {
        for ( unsigned int i = 58; i < ctc_string.size() - 1; i += 68 )
        {
            ctc_string.insert( i, "\"//\n     &\"" );
        }
    }
    else if ( form == F_FREE )
    {
        for ( unsigned int i = 58; i < ctc_string.size() - 1; i += 68 )
        {
            ctc_string.insert( i, "\"//&\n      \"" );
        }
    }

    return ctc_string;
}

/* *INDENT-OFF* */
/**
 * This function writes the definitions of the region handle and the
 * CTC-string for the region referred to by this directive object.
 *
 * Fortran 90 example:
 * @code

      INTEGER( KIND=8 ) :: opari2_region_1
      CHARACTER (LEN=268), parameter :: opari2_ctc_1 =&
      "265*regionType=parallel*sscl=/some/path/to/source.f90:26"//
      ":26*escl=/some/path/to/source.f90:43:43*has_if=1*has_num"//
      "_threads=4*has_reduction=1**"

 * @endcode
 *
 * C/C++ example:
 * @code

   static OPARI2_Region_handle opari2_region_1 = NULL;
   #define opari2_ctc_1 "261*regionType=parallel*sscl=/some/path/to/source.c:35:35*escl=/some/path/to/source.c:54:54*has_if=1*has_num_threads=4*has_reduction=1**"

 * @endcode
 */
/* *INDENT-ON* */
void
OPARI2_Directive::generate_descr_common( ostream& os  )
{
    string ctc_string = generate_ctc_string( s_format );

    if ( s_lang & L_F77 )
    {
        os << "      INTEGER*8 " << region_id_prefix << m_id << "\n";
        os << "      CHARACTER*" << m_ctc_string_len << " " << m_ctc_string_variable << "\n";
        if ( s_format == F_FIX )
        {
            os << "      PARAMETER (" << m_ctc_string_variable << "=\n";
            os << "     &" << ctc_string << ")\n\n";
        }
        else
        {
            os << "      PARAMETER (" << m_ctc_string_variable << "=&\n";
            os << "      " << ctc_string << ")\n\n";
        }

        s_common_block.push_back( m_id );
    }
    else if ( s_lang & L_F90 )
    {
        os << "      INTEGER( KIND=8 ) :: " << region_id_prefix << m_id << "\n\n";
        if ( s_format == F_FREE )
        {
            os << "      CHARACTER (LEN=" << m_ctc_string_len << "), parameter :: ";
            os << m_ctc_string_variable << " =&\n      " << ctc_string << "\n\n";
        }
        else
        {
            os << "      CHARACTER (LEN=" << m_ctc_string_len << "), parameter :: ";
            os << m_ctc_string_variable << " =\n     &" << ctc_string << "\n\n";
        }

        s_common_block.push_back( m_id );
    }
    else if ( s_lang & L_C_OR_CXX )
    {
        os << "static " << "OPARI2_Region_handle "
           << region_id_prefix << m_id;

        if ( s_preprocessed_file )
        {
            os << " = (OPARI2_Region_handle)0;\n";
        }
        else
        {
            os << " = NULL;\n";
        }
        os << "#define " << m_ctc_string_variable << " " << ctc_string << "\n";
    }
}


void
OPARI2_Directive::remove_commas( void )
{
    int bracket_counter = 0;

    for ( unsigned int line = 0; line < m_lines.size(); line++ )
    {
        for ( unsigned int c = 0; c < m_lines[ line ].length(); c++ )
        {
            if ( m_lines[ line ][ c ] == '(' )
            {
                bracket_counter++;
            }
            if ( m_lines[ line ][ c ] == ')' )
            {
                bracket_counter--;
            }
            if ( bracket_counter == 0 && m_lines[ line ][ c ] == ',' )
            {
                m_lines[ line ][ c ] = ' ';
            }
        }
    }

    return;
}


/** @brief Returns the arguments of a clause. */
string
OPARI2_Directive::find_arguments( unsigned&          line,
                                  string::size_type& pos,
                                  bool               remove,
                                  string             clause )
{
    string arguments;
    int    bracket_counter = 0;

    if ( remove )
    {
        m_lines[ line ].replace( pos, clause.length(), string( clause.length(), ' ' ) );
    }
    pos += clause.length();

    pos = m_lines[ line ].find_first_not_of( " \t", pos );
    if ( ( pos != string::npos ) && ( m_lines[ line ][ pos ] == '(' ) )
    {
        bracket_counter++;
        if ( remove )
        {
            m_lines[ line ][ pos ] = ' ';
        }
        pos++;
    }

    bool contComm = false;       // Continuation line or comment found

    while ( bracket_counter > 0 )
    {
        if ( m_lines[ line ][ pos ] == '(' )
        {
            bracket_counter++;
        }
        if ( ( s_lang & L_C_OR_CXX && m_lines[ line ][ pos ] == '\\' ) ||
             ( s_lang & L_FORTRAN  && m_lines[ line ][ pos ] == '&' ) )
        {
            pos = m_lines[ line ].length();
        }
        else
        {
            if ( s_lang & L_FORTRAN && m_lines[ line ][ pos ] == '!' )
            {
                contComm = true;
            }
            else
            {
                arguments.append( 1, m_lines[ line ][ pos ] );
                if ( remove )
                {
                    m_lines[ line ][ pos ] = ' ';
                }
                pos++;
            }
        }

        if ( pos >= m_lines[ line ].length() || contComm )
        {
            if ( s_lang & L_FORTRAN && !remove_empty_line( line ) )
            {
                line++;
            }
            contComm = false;

            if ( line >= m_lines.size() )
            {
                std::cerr << m_filename << ":" << m_begin_first_line << ": ERROR: Missing ) for "
                          << clause << " clause \n" << std::endl;
                cleanup_and_exit();
            }
            else if ( s_lang & L_FORTRAN )
            {
                pos = m_lines[ line ].find_first_of( "!*cC" ) + 6;
                pos = m_lines[ line ].find_first_not_of( " \t", pos );
                if ( m_lines[ line ][ pos ] == '&' )
                {
                    pos++;
                }
            }
            else if ( s_lang & L_C_OR_CXX )
            {
                pos = 0;
            }
        }

        if (  m_lines[ line ][ pos ] == ')' )
        {
            bracket_counter--;
        }
    }

    //remove last bracket if necessary
    if ( remove && pos != string::npos )
    {
        m_lines[ line ][ pos ] = ' ';
        //remove comma after the removed clause if needed
        while ( m_lines[ line ][ pos ] == ' ' )
        {
            pos++;
        }
        if ( m_lines[ line ][ pos ] == ',' )
        {
            m_lines[ line ][ pos ] =  ' ';
        }
    }


    size_t p;
    p = arguments.find( ' ' );
    while ( p != string::npos )
    {
        arguments.erase( p, 1 );
        p = arguments.find( ' ' );
    }
    return arguments;
}


/** remove empty continued lines inside of pragma*/
bool
OPARI2_Directive::remove_empty_line( unsigned& line )
{
    string lline = m_lines[ line ];
    lline.erase( remove_if( lline.begin(), lline.end(), isspace ), lline.end() );
    transform( lline.begin(), lline.end(), lline.begin(), tolower );
    if ( lline.compare( m_directive_prefix[ 0 ] ) == 0 )
    {
        if ( m_lines.size() == ( line + 1 ) )
        {
            m_lines[ line - 1 ][ m_lines[ line - 1 ].find_last_of( "&" ) ] = ' ';
        }
        m_lines.erase( m_lines.begin() + line );
        return true;
    }
    return false;
}


bool
OPARI2_Directive::find_word( const string       word,
                             unsigned&          line,
                             string::size_type& pos )
{
    if ( s_lang & L_C_OR_CXX )
    {
        for ( unsigned i = line; i < m_lines.size(); ++i )
        {
            string::size_type w = m_lines[ i ].find( word );
            while ( w != string::npos )
            {
                char a;
                char b;
                //word may start at position 0 of a continuation line
                if ( w == 0 )
                {
                    b = ' ';
                }
                else
                {
                    b = m_lines[ i ][ w - 1 ];
                }

                if ( m_lines[ i ].length() > w + word.length() )
                {
                    a = m_lines[ i ][ w + word.length() ];
                }
                else
                {
                    a = ' ';
                }

                if ( ( b == ' ' || b == '\t' || b == '/' || b == ')' || b == ',' || b == '#' ) &&
                     ( a == ' ' || a == '\t' || a == '/' || a == '(' || a == ',' ) )
                {
                    line = i;
                    pos  = w;
                    return true;
                }
                else
                {
                    w++;
                    if ( w != string::npos )
                    {
                        w = m_lines[ i ].find( word, w );
                    }
                }
            }
            pos = 0;
        }
    }
    else if ( s_lang & L_FORTRAN )
    {
        for ( unsigned i = line; i < m_lines.size(); ++i )
        {
            string::size_type s = ( pos == 0 ) ? m_lines[ i ].find( m_directive_prefix[ 0 ] ) + m_directive_prefix[ 0 ].length() : pos;
            string::size_type w = m_lines[ i ].find( word, s );
            string::size_type c = m_lines[ i ].find( '!', s );
            // if word found and found before comment
            while ( w != string::npos &&
                    ( c == string::npos || ( c != string::npos && w < c ) )
                    )
            {
                char b = m_lines[ i ][ w - 1 ];
                char a;
                if ( m_lines[ i ].length() > w + word.length() )
                {
                    a = m_lines[ i ][ w + word.length() ];
                }
                else
                {
                    a = ' ';
                }
                if ( ( b == ' ' || b == '\t' || b == '!' || b == ')' || b == ',' ) &&
                     ( a == ' ' || a == '\t' || a == '!' || a == '(' || a == ',' || a == '&' ) )
                {
                    line = i;
                    pos  = w;
                    return true;
                }
                else
                {
                    w++;
                    if ( w != string::npos )
                    {
                        w = m_lines[ i ].find( word, w );
                    }
                }
            }
            pos = 0;
        }
    }
    return false;
}


string
OPARI2_Directive::find_next_word( void )
{
    if ( s_lang & L_C_OR_CXX )
    {
        while ( m_pline < m_lines.size() )
        {
            string::size_type wbeg = m_lines[ m_pline ].find_first_not_of( " \t", m_ppos );
            if ( wbeg == string::npos || m_lines[ m_pline ][ wbeg ] == '\\' )
            {
                ++m_pline;
                if ( m_pline < m_lines.size() )
                {
                    m_ppos = 0;
                }
                else
                {
                    return "";
                }
            }
            else if ( m_lines[ m_pline ][ wbeg ] == '(' ||
                      m_lines[ m_pline ][ wbeg ] == ')' ||
                      m_lines[ m_pline ][ wbeg ] == '#' )
            {
                m_ppos = wbeg + 1;
                return string( 1, m_lines[ m_pline ][ wbeg ] );
            }
            else
            {
                m_ppos = m_lines[ m_pline ].find_first_of( " \t()", wbeg );
                return m_lines[ m_pline ].substr( wbeg, m_ppos == string::npos ? m_ppos : m_ppos - wbeg );
            }
        }
    }
    else if ( s_lang & L_FORTRAN )
    {
        string sentinel = m_directive_prefix[ 0 ];
        while ( m_pline < m_lines.size() )
        {
            string::size_type wbeg = m_lines[ m_pline ].find_first_not_of( " \t", m_ppos );
            if ( wbeg == string::npos || m_lines[ m_pline ][ wbeg ] == '&' )
            {
                ++m_pline;
                if ( m_pline < m_lines.size() )
                {
                    m_ppos = m_lines[ m_pline ].find( sentinel ) + sentinel.length();
                    m_ppos = m_lines[ m_pline ].find_first_not_of( " \t", m_ppos );
                    if ( m_lines[ m_pline ][ m_ppos ] == '&' ||
                         m_lines[ m_pline ][ m_ppos ] == '+' )
                    {
                        ++m_ppos;
                    }
                }
                else
                {
                    return "";
                }
            }
            else if ( m_lines[ m_pline ][ wbeg ] == '(' || m_lines[ m_pline ][ wbeg ] == ')' )
            {
                m_ppos = wbeg + 1;
                return string( 1, m_lines[ m_pline ][ wbeg ] );
            }
            else
            {
                m_ppos = m_lines[ m_pline ].find_first_of( " \t()&", wbeg );
                return m_lines[ m_pline ].substr( wbeg, m_ppos == string::npos ? m_ppos : m_ppos - wbeg );
            }
        }
    }
    return "";
}


void
OPARI2_Directive::fix_clause_arg( vector<string>&    outer,
                                  vector<string>&    inner,
                                  unsigned&          line,
                                  string::size_type& pos )
{
    char* optr = &( outer[ line ][ pos ] );
    char* iptr = &( inner[ line ][ pos ] );

    string            sentinel = m_directive_prefix[ 0 ];
    string::size_type slen     = sentinel.length();

    while ( *optr != ')' )
    {
        while ( ( ( s_lang & L_C_OR_CXX ) &&  *optr == '\\' )                 ||
                ( ( ( s_lang & L_FORTRAN  ) && ( *optr == '!' || *optr == '&' ) ) ||
                  pos >= outer[ line ].size() ) )
        {
            // skip to next line
            ++line;
            if ( line >= outer.size() )
            {
                return;
            }

            if ( s_lang & L_FORTRAN )
            {
                pos = outer[ line ].find( sentinel ) + slen;
                pos = outer[ line ].find_first_not_of( " \t", pos );
                if ( outer[ line ][ pos ] == '&' )
                {
                    ++pos;
                }
            }
            else if ( s_lang & L_C_OR_CXX )
            {
                pos = 0;
            }

            optr = &( outer[ line ][ pos ] );
            iptr = &( inner[ line ][ pos ] );
        }
        *iptr = *optr;
        *optr = ' ';
        ++iptr;
        ++optr;
        ++pos;
    }

    *iptr = ')';
    *optr = ' ';
}


/**
 * @brief Remove empty lines in directive 'lines'.
 */
void
OPARI2_Directive::remove_empties( void )
{
    if ( s_lang & L_C_OR_CXX )
    {
        vector<string>::iterator it = m_lines.begin();
        while ( it != m_lines.end() )
        {
            string::size_type l = it->find_first_not_of( " \t&" );
            if ( l == string::npos || ( *it )[ l ] == '\\' )
            {
                it = m_lines.erase( it );
            }
            else
            {
                ++it;
            }
        }

        // make sure last line is not a continued line
        int               lastline = m_lines.size() - 1;
        string::size_type lastpos  = m_lines[ lastline ].size() - 1;
        if ( m_lines[ lastline ][ lastpos ] == '\\' )
        {
            m_lines[ lastline ][ lastpos ] = ' ';
        }
    }
    else if ( s_lang & L_FORTRAN )
    {
        // remove lines without content
        string                   sentinel = m_directive_prefix[ 0 ];
        vector<string>::iterator it       = m_lines.begin();
        while ( it != m_lines.end() )
        {
            string::size_type pos = it->find( sentinel ) + sentinel.length();
            if ( ( *it )[ pos ] == '&' || ( *it )[ pos ] == '+' )
            {
                ++pos;
            }
            pos = it->find_first_not_of( " \t&", pos );
            if ( pos == string::npos || ( *it )[ pos ] == '!' )
            {
                it = m_lines.erase( it );
            }
            else
            {
                ++it;
            }
        }

        // make sure 1st line is not a continuation line
        string::size_type pos = m_lines[ 0 ].find( sentinel );
        if ( pos == 0 )
        {
            m_lines[ 0 ][ sentinel.length() ] = ' ';
        }
        else
        {
            string::size_type l = m_lines[ 0 ].find_first_not_of( " \t", pos + sentinel.length() );
            if ( m_lines[ 0 ][ l ] == '&' || m_lines[ 0 ][ l ] == '+' )
            {
                m_lines[ 0 ][ l ] = ' ';
            }
        }

        // make sure last line is not a continued line
        int lastline = m_lines.size() - 1;
        pos = m_lines[ lastline ].find( sentinel ) + sentinel.length();
        pos = m_lines[ lastline ].find( '!', pos );
        if ( pos != string::npos )
        {
            --pos;
        }
        string::size_type amp = m_lines[ lastline ].find_last_not_of( " \t", pos );
        if ( m_lines[ lastline ][ amp ] == '&' )
        {
            m_lines[ lastline ][ amp ] = ' ';
        }
    }
}


/** Initialize static variables */

void
OPARI2_Directive::SetOptions(  OPARI2_Language_t lang,
                               OPARI2_Format_t   form,
                               bool              keep_src,
                               bool              preprocessed,
                               const string      id )
{
    s_lang                 = lang;
    s_format               = form;
    s_keep_src_info        = keep_src;
    s_preprocessed_file    = preprocessed;
    s_inode_compiletime_id = id;
}

/**
 * Where/when are they modified?
 */
vector<int>       OPARI2_Directive::      s_common_block;
OPARI2_Directive* OPARI2_Directive::s_outer                         = NULL;
int               OPARI2_Directive::              s_num_all_regions = 0;
string            OPARI2_Directive::           s_inode_compiletime_id;
OPARI2_Language_t OPARI2_Directive::s_lang                           = L_NA;
OPARI2_Format_t   OPARI2_Directive::  s_format                       = F_NA;
bool              OPARI2_Directive::             s_keep_src_info     = false;
bool              OPARI2_Directive::             s_preprocessed_file = false;
