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
 * Copyright (c) 2009-2013, 2016,
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
 *  @file      opari2_parser_c.cc
 *
 *  @brief     This file contains all functions to parse and process C or C++ files.
 */

#include <config.h>
#include <iostream>
using std::istream;
using std::cerr;
#include <stack>
using std::stack;
#include <vector>
using std::vector;
#include <map>
using std::map;
#include <string>
using std::string;
#include <iostream>
using std::getline;
#include <cctype>
using std::isalnum;
using std::isalpha;
#include <cstdlib>
#include <cassert>

#include "opari2.h"
#include "opari2_parser_c.h"
#include "openmp/opari2_directive_openmp.h"
#include "pomp/opari2_directive_pomp.h"
#include "offload/opari2_directive_offload.h"
#include "opari2_directive_manager.h"


OPARI2_CParser::OPARI2_CParser( OPARI2_Option_t& options )
    : m_options( options ), m_os( options.os ), m_is( options.is )
{
    m_line          = "";
    m_pos           = 0;
    m_in_comment    = false;
    m_in_string     = false;
    m_pre_cont_line = false;
    m_require_end   = true;
    m_is_for        = false;
    m_block_closed  = false;
    m_lineno        = 0;
    m_level         = 0;
    m_num_semi      = 0;
    m_lstart        = string::npos;

    m_current_file = options.infile;
    m_infile       = options.infile;

    m_next_end.push( -1 );
}

string
OPARI2_CParser::find_next_word( unsigned&          pline,
                                string::size_type& ppos )
{
    unsigned size = m_pre_stmt.size();
    while ( pline < size )
    {
        string::size_type wbeg = m_pre_stmt[ pline ].find_first_not_of( " \t", ppos );
        if ( m_pre_stmt[ pline ][ wbeg ] == '\\' || wbeg == string::npos )
        {
            ++pline;
            if ( pline < size )
            {
                ppos = 0;
            }
            else
            {
                return "";
            }
        }
        else
        {
            ppos = m_pre_stmt[ pline ].find_first_of( " \t()", wbeg );
            return m_pre_stmt[ pline ].substr( wbeg,
                                               ppos == string::npos ? ppos : ppos - wbeg );
        }
    }
    return "";
}


/** @brief Check whether the current line is an extern function
    declaration */
bool
OPARI2_CParser::is_extern_decl( void )
{
    // If it starts with 'extern' it is an declaration
    int nrstart = m_line.find_first_not_of( " \t" );
    int nrend   = m_line.find_first_of( " \t", nrstart + 1 );
    if ( nrend > nrstart && m_line.substr( nrstart, nrend ) == "extern" )
    {
        return true;
    }
    return false;
}



/**
 * @brief  Instrument pragma directives.
 *
 * Preprocessor lines are passed to this function and checked, whether they are
 * pragmas.
 */
bool
OPARI2_CParser::process_prestmt( int               ln,
                                 string::size_type ppos )
{
    unsigned s          = m_pre_stmt.size();
    bool     in_comment = false;

    vector<string> orig_stmt;

    /* "remove" comments */
    for ( unsigned i = 0; i < s; ++i )
    {
        string::size_type pos  = 0;
        string&           line = m_pre_stmt[ i ];
        string            orig_line( line );

        orig_stmt.push_back( orig_line );

        while ( pos < line.size() )
        {
            if ( in_comment )
            {
                /* look for comment end */
                if ( line[ pos ] == '*' && line[ pos + 1 ] == '/' )
                {
                    line[ pos++ ] = ' ';
                    in_comment    = false;
                }
                line[ pos++ ] = ' ';
            }
            else if ( line[ pos ] == '/' )
            {
                pos++;
                if ( line[ pos ] == '/' )
                {
                    // c++ comments /
                    line[ pos - 1 ] = ' ';
                    line[ pos++ ]   = ' ';
                    while ( pos < line.size() )
                    {
                        line[ pos++ ] = ' ';
                    }
                }
                else if ( line[ pos ] == '*' )
                {
                    // c comment start
                    line[ pos - 1 ] = ' ';
                    line[ pos++ ]   = ' ';
                    in_comment      = true;
                }
            }
            else
            {
                pos++;
            }
        }
    }

    vector<string> directive_prefix( 1, "#" );
    unsigned       pline = 0;
    string         first = find_next_word( pline, ppos );

    if ( first == "pragma" )
    {
        directive_prefix.push_back( first );

        string paradigm_identifier = find_next_word( pline, ppos );
        directive_prefix.push_back( paradigm_identifier );

        OPARI2_Directive* d =
            NewDirective( m_pre_stmt, directive_prefix, m_options.lang,
                          m_current_file, m_lineno - m_pre_stmt.size() + 1 );

        if ( d )
        {
            ProcessDirective( d, m_os, &m_require_end, &m_is_for );
            return true;
        }
    }
    else if ( first == "include" )
    {
        // include <xxx.h> (for supported runtime API) -> remove it
        string word = find_next_word( pline, ppos );
        if ( IsSupportedAPIHeaderFile( word, m_options.lang ) )
        {
            s = 0;
        }
    }

    for ( unsigned i = 0; i < s; ++i )
    {
        m_os << orig_stmt[ i ] << "\n";
    }

    orig_stmt.clear();

    return false;
}


bool
OPARI2_CParser::handle_closed_block()
{
    bool   newline_printed = false;
    size_t next_char       = m_line.find_first_not_of( " \t", m_pos );
    /* The treatment of else blocks doesn't work yet, but it should go
     * somewhere here */
    if ( ( m_next_end.top() == m_level         &&
           ( next_char != string::npos         &&
             m_line.substr( next_char, 4 ) != "else" ) ) ||
         next_char == string::npos  )
    {
        bool more_chars = ( m_pos < m_line.size() );
        if ( !m_block_closed )
        {
            m_os << '\n';
            newline_printed = true;
        }
        // while because block can actually close more than one pragma
        while ( m_next_end.top() == m_level )
        {
            OPARI2_Directive* d_top = DirectiveStackTop( NULL );
            if ( d_top )
            {
                vector<string> directive_prefix;
                int            l = m_lineno;
                if ( m_block_closed )
                {
                    l--;
                }

                OPARI2_Directive* d =
                    NewDirective( m_end_stmt, directive_prefix, m_options.lang,
                                  m_current_file, l );
                if ( d )
                {
                    ProcessDirective( d, m_os );
                }
                else
                {
                    cerr << "Error when handling end of directive:" << d_top->GetName() << "\n";
                }
            }
            m_next_end.pop();
        }

        if ( more_chars && !m_block_closed )
        {
            if ( next_char != string::npos )
            {
                m_os << "#line " << m_lineno << " \"" << m_current_file << "\"" << "\n";
            }

            for ( unsigned i = 0; i < m_pos; ++i )
            {
                m_os << ' ';
            }
        }
    }

    return newline_printed;
}


void
OPARI2_CParser::handle_preprocessor_directive( void )
{
    if ( ( isdigit( m_line[ m_line.find_first_not_of( " \t", m_lstart + 1 ) ] ) ) )
    {
        /**
         * which compiler supports the line directive in this form?
         */
        int    nrstart, nrend;
        string filename;
        nrstart = m_line.find_first_not_of( " \t", m_lstart + 1 );
        nrend   = m_line.find_first_of( " \t", nrstart + 1 );
        // -1 because the next line has the number specified in this directive
        m_lineno = atoi( m_line.substr( nrstart, nrend ).c_str() ) - 1;
        nrstart  = m_line.find_first_not_of( " \t\"", nrend + 1 );
        nrend    = m_line.find_first_of( " \t\"", nrstart + 1 );
        filename = m_line.substr( nrstart, nrend - nrstart );
        if ( nrstart != 0 )
        {
            if ( filename[ 0 ] == '/' )
            {
                /*absolute path*/
                m_current_file = filename;
            }
            else
            {
                /*relative path*/
                string path( m_infile );
                path           = path.substr( 0, path.find_last_of( "/" ) );
                m_current_file = path + "/" + filename;
            }
        }
        m_os << m_line << std::endl;
    }
    else if ( m_line.compare( m_lstart + 1, 5, "line " ) == 0 &&
              isdigit( m_line[ m_line.find_first_not_of( " \t", m_lstart + 5 ) ] ) )
    {
        int    nrstart, nrend;
        string filename;
        nrstart = m_line.find_first_not_of( " \t", m_lstart + 5 );
        nrend   = m_line.find_first_of( " \t", nrstart + 1 );
        // -1 because the next line has the number specified in this directive
        m_lineno = atoi( m_line.substr( nrstart, nrend ).c_str() ) - 1;
        nrstart  = m_line.find_first_not_of( " \t\"", nrend + 1 );
        nrend    = m_line.find_first_of( " \t\"", nrstart + 1 );
        if ( nrstart != 0 )
        {
            filename = m_line.substr( nrstart, nrend - nrstart );
            if ( filename[ 0 ] == '/' )
            {
                /* absolute path */
                m_current_file = filename;
            }
            else
            {
                /* relative path */
                string path( m_infile );
                path           = path.substr( 0, path.find_last_of( "/" ) );
                m_current_file = path + "/" + filename;
            }
        }
        m_os << m_line << std::endl;
    }
    else
    {
        /*
         * other preprocessor directive
         */
        m_pre_stmt.push_back( m_line );
        if ( m_line[ m_line.size() - 1 ] == '\\' ) // escaped Backslash
        {
            m_pre_cont_line = true;
        }
        else
        {
            if ( process_prestmt( m_lineno, m_lstart + 1 ) )
            {
                if ( m_require_end )
                {
                    m_next_end.push( m_level );
                    m_num_semi = m_is_for ? 3 : 1;
                }
                else
                {
                    m_num_semi = 0;
                }
            }
            m_pre_stmt.clear();
        }
    }
}


void
OPARI2_CParser::handle_preprocessor_continuation_line( void )
{
    m_pre_stmt.push_back( m_line );
    /* check for multiline comments in preprocessor directives */
    if ( !m_in_comment && m_line.find( "/*" ) != string::npos &&
         !( m_line.find( "*/" ) != string::npos &&
            m_line.find( "/*" ) < m_line.find( "*/" ) ) )
    {
        m_in_comment = true;
    }
    else if ( m_in_comment && m_line.find( "*/" ) != string::npos )
    {
        m_in_comment = false;
    }

    if ( m_line[ m_line.size() - 1 ] != '\\' && !m_in_comment )
    {
        m_pre_cont_line = false;
        if ( process_prestmt( m_lineno - m_pre_stmt.size() + 1, m_lstart + 1 ) )
        {
            if ( m_require_end )
            {
                m_next_end.push( m_level );
                m_num_semi = m_is_for ? 3 : 1;
            }
            else
            {
                m_num_semi = 0;
            }
        }
        m_pre_stmt.clear();
    }
}


void
OPARI2_CParser::handle_regular_line()
{
    bool newline_printed = false;

    while ( m_pos < m_line.size() )
    {
        newline_printed = false;
        if ( m_in_comment )
        {
            // look for comment end
            if ( m_line[ m_pos ] == '*' && m_line[ m_pos + 1 ] == '/' )
            {
                m_os << "*/";
                m_in_comment = false;
                m_pos       += 2;
            }
            else
            {
                m_os << m_line[ m_pos++ ];
            }
        }
        else if ( m_line[ m_pos ] == '/' )
        {
            m_pos++;
            if ( m_line[ m_pos ] == '/' )
            {
                // c++ comments
                m_pos++;
                m_os << "//";
                while ( m_pos < m_line.size() )
                {
                    m_os << m_line[ m_pos++ ];
                }
            }
            else if ( m_line[ m_pos ] == '*' )
            {
                // c comment start
                m_pos++;
                m_os << "/*";
                m_in_comment = true;
            }
            else
            {
                m_os << '/';
            }
        }
        else if ( m_in_string || m_line[ m_pos ] == '\"' )
        {
            // character string constant
            if ( m_in_string )
            {
                m_in_string = false;
                m_pos--; // to make sure current character gets reprocessed
            }
            else
            {
                m_os << "\"";
            }
            m_pos++;
            while ( m_line[ m_pos ] != '\"' )
            {
                if ( m_line[ m_pos ] == '\\' )
                {
                    m_os << '\\';
                    m_pos++;
                    if ( m_line[ m_pos ] == '\0' )
                    {
                        m_in_string = true;
                        break;
                    }
                    else if ( m_line[ m_pos ] == '\\' )
                    {
                        if ( m_line[ m_pos + 1 ] == '\0' )
                        {
                            m_os << '\\';
                            m_in_string = true;
                            break;
                        }
                    }
                }
                m_os << m_line[ m_pos ];
                m_pos++;
            }
            if ( !m_in_string )
            {
                m_os << '\"';
            }
            m_pos++;
        }
        else if ( m_line[ m_pos ] == '\'' )
        {
            // character constant
            m_os << "\'";
            do
            {
                m_pos++;
                if ( m_line[ m_pos ] == '\\' )
                {
                    m_os << '\\';
                    m_pos++;
                    if ( m_line[ m_pos ] == '\'' )
                    {
                        m_os << '\'';
                        m_pos++;
                    }
                }
                m_os << m_line[ m_pos ];
            }
            while ( m_line[ m_pos ] != '\'' );
            m_pos++;
        }
        else if ( isalpha( m_line[ m_pos ] ) || m_line[ m_pos ] == '_' )
        {
            // identifier
            string::size_type startpos = m_pos;
            while ( m_pos < m_line.size() &&
                    ( isalnum( m_line[ m_pos ] ) || m_line[ m_pos ] == '_' ) )
            {
                m_pos++;
            }
            string ident( m_line, startpos, m_pos - startpos );

            /* Replace if valid runtime function */
            ReplaceRuntimeAPI( ident, ident, m_current_file, L_C_OR_CXX );
            m_os << ident;

            if ( ident == "for" && m_num_semi == 1 )
            {
                m_num_semi = 3;
            }
        }
        else if ( m_line[ m_pos ] == '{' )
        {
            // block open
            m_os << m_line[ m_pos++ ];
            m_level++;
            m_num_semi = 0;
        }
        else if ( m_line[ m_pos ] == '}' )
        {
            // block close
            m_os << m_line[ m_pos++ ];
            m_level--;
            size_t next_char = m_line.find_first_not_of( " \t", m_pos );
            if ( next_char != string::npos )
            {
                newline_printed = handle_closed_block();
            }
            else
            {
                m_block_closed = true;
            }
        }
        else if ( m_line[ m_pos ] == ';' )
        {
            // statement end
            m_os << m_line[ m_pos++ ];
            m_num_semi--;
            if ( m_num_semi == 0 )
            {
                //newline_printed = handle_closed_block();
                m_block_closed = true;
            }
        }
        else
        {
            m_os << m_line[ m_pos++ ];
        }
    }
    if ( !newline_printed )
    {
        m_os << '\n';
    }
}

bool
OPARI2_CParser::get_next_line( void )
{
    bool success = ( bool )getline( m_is, m_line );
    ++m_lineno;
    m_pos = 0;

    //std::cout << m_lineno << ": " << m_line << std::endl;

    if ( success )
    {
        /* workaround for bogus getline implementations */
        while ( m_line.size() == 1 && m_line[ 0 ] == '\0' )
        {
            success = ( bool )getline( m_is, m_line );
            ++m_lineno;
        }

        /* remove extra \r from Windows source files */
        if ( m_line.size() && *( m_line.end() - 1 ) == '\r' )
        {
            m_line.erase( m_line.end() - 1 );
        }
    }

    return success;
}


/** Parse source file line by line, search for directives and the
 *  related code blocks. Comments and strings are removed to avoid
 *  finding keywords in comments.
 */
void
OPARI2_CParser::process( void )
{
    while ( get_next_line() )
    {
        string::size_type ls;
        if ( m_block_closed   &&
             !( ( ( m_line.size() - m_line.find_first_not_of( " \t" ) ) > 3 ) &&
                (  m_line.find_first_not_of( " \t" ) != string::npos ) &&
                m_line.substr( m_line.find_first_not_of( " \t" ), 4 ) == "else" ) )
        {
            handle_closed_block();
            m_block_closed = false;
        }

        /* start offload region if __declspec is found and continue to
         * parse the rest of the line as usual without the __declspec
         */
        if ( !m_in_comment &&
             ( ( ls = m_line.find_first_not_of( " \t" ) ) != string::npos ) &&
             m_line.substr( ls, 10 ) == "__declspec" )
        {
            m_pre_stmt.push_back( m_line );
            vector<string>    directive_prefix( 1, "__declspec" );
            OPARI2_Directive* d =
                NewDirective( m_pre_stmt, directive_prefix, m_options.lang,
                              m_current_file, m_lineno );
            assert( d );

            ProcessDirective( d, m_os );

            m_pre_stmt.clear();
            m_next_end.push( m_level );
            int               cl_brackets = 0;
            string::size_type pos         = 0;
            while ( cl_brackets < 2 && pos < m_line.size() )
            {
                if ( m_line[ pos ] == ')' )
                {
                    cl_brackets++;
                }
                m_os << m_line[ pos ];
                m_line[ pos ] = ' ';
                pos++;
            }
            m_os << "\n";
        }

        if ( m_pre_cont_line )
        {
            handle_preprocessor_continuation_line();
        }
        else if ( !m_in_comment && m_options.preprocessed_file &&
                  ( m_line == "___POMP2_INCLUDE___"  ||
                    m_line == "___POMP2_INCLUDE___ " ) ) // Studio compiler appends a blank during preprocessing
        {
            m_os << "#include \"" << m_options.incfile << "\"" << "\n";
        }
        else if ( !m_in_comment &&
                  ( ( m_lstart = m_line.find_first_not_of( " \t" ) ) != string::npos ) &&
                  m_line[ m_lstart ] == '#' )
        {
            handle_preprocessor_directive();
        }
        else
        {
            handle_regular_line();
        }
    }

    if ( m_block_closed )
    {
        handle_closed_block();
        m_block_closed = false;
    }
    // check end position stack
    if ( m_next_end.top() != -1 )
    {
        cerr << "ERROR: could not determine end of OpenMP construct (braces mismatch?)\n";
        PrintDirectiveStackTop();
        cleanup_and_exit();
    }
}
