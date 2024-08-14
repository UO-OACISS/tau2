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
 * Copyright (c) 2009-2013, 2014, 2016,
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
 *  @file       opari2_parser_f.cc
 *
 *  @brief      All functions to parse fortran files are collected here.
 */


#include <config.h>
#include <cctype>
using std::tolower;
using std::toupper;
#include <string>
using std::string;
#include <iostream>
using std::getline;
#include <algorithm>
using std::transform;
using std::sort;
using std::min;
#include <functional>
using std::greater;
#include <cstring>
using std::strlen;
using std::remove_if;

#include "config.h"
#include "opari2_parser_f.h"
#include "opari2_directive_manager.h"

struct fo_tolower : public std::unary_function<int, int>
{
    int
    operator()( int x ) const
    {
        return std::tolower( x );
    }
};

OPARI2_FortranParser::OPARI2_FortranParser( OPARI2_Option_t& options )
    : m_options( options ), m_os( options.os ), m_is( options.is )
{
    m_line             = "";
    m_lowline          = "";
    m_unprocessed_line = "";
    m_lineno           = 0;
    //m_loop_directives   = NULL;
    m_need_pragma  = false;
    m_in_string    = 0;
    m_normal_line  = false;
    m_in_header    = false;
    m_continuation = false;
    m_sentinel     = "";

    m_offload_pragma           = false;
    m_offload_attribute        = "";
    m_current_offload_function = "";
    m_offload_subroutine       = false;
    m_offload_function         = false;

    m_waitfor_loopstart = false;
    m_waitfor_loopend   = false;
    m_lineno_loopend    = 0;

    m_curr_file = m_options.infile;

    if ( m_options.keep_src_info )
    {
        /** The preprocessor of the Sun Studio compiler breaks if the
         *  first line is a preprocessor directive, so we insert a
         *  blank line at the beginning */
        m_os << std::endl;
        m_os << "#line 1 \"" << m_options.infile << "\"" << "\n";
    }
}


/**@brief Check if the line belongs to the header of a subroutine or function.
 *        After lines in the header, we can insert our variable definitions.*/
bool
OPARI2_FortranParser::is_sub_unit_header( void )
{
    string      sline;
    string      lline;
    string      keyword;
    bool        result;
    static bool continuation = false;
    static int  openbrackets = 0;
    static bool inProgram    = false;
    static bool inModule     = false;
    static bool inInterface  = false;
    static bool inContains   = false;

    size_t pos;

    pos = m_lowline.find_first_not_of( " \t" );
    /*string is empty*/

    if ( pos == string::npos )
    {
        pos = 0;
        sline.clear();
    }
    else
    {
        sline = m_lowline.substr( pos );
    }
    lline = sline;
    sline.erase( remove_if( sline.begin(), sline.end(), isspace ), sline.end() );

    //Set number of open brackets to 0 if new unit begins, since we might have missed
    //closing brackets on continuation lines.
    if ( ( ( sline.find( "program" ) == 0 )  && lline.find( "program" ) != string::npos )               ||
         ( ( sline.find( "module" ) == 0 ) && !inProgram && lline.find( "module" ) != string::npos )     ||
         ( ( sline.find( "interface" ) == 0 ) && inModule && lline.find( "interface" ) != string::npos ) ||
         ( ( sline.find( "abstractinterface" ) == 0 ) && inModule )  ||
         ( ( sline.find( "contains" ) != string::npos ) && inModule  && lline.find( "contains" ) != string::npos )  ||
         ( sline.find( "subroutine" ) == 0  && lline.find( "subroutine" ) != string::npos )                                                 ||
         ( ( sline.find( "function" ) == 0  && lline.find( "function" ) != string::npos )  &&
           !m_in_header                                &&
           ( ( sline.find( "=" ) >= sline.find( "!" ) )             ||
             ( sline.find( "=" ) >= sline.find( "kind" ) ) ) ) )
    {
        openbrackets = 0;
    }

    //Check if we are in Fortran77 and have a character in column 6
    if ( ( m_options.form & F_FIX ) && m_lowline.length() >= 6 && m_lowline[ 5 ] != ' ' && m_lowline.find( "\t" ) > 6 )
    {
        continuation = true;
    }

    //Check if we enter a program block
    inProgram = inProgram || ( sline.find( "program" ) == 0 );
    //Check if we enter a module block
    inModule = !inProgram && ( inModule || ( sline.find( "module" ) == 0 ) );
    //Check if we enter an interface block
    inInterface = inModule && ( inInterface || ( sline.find( "interface" ) == 0 ) || ( sline.find( "abstractinterface" ) == 0 ) );
    //Check if we enter a contains block
    inContains = inModule && ( inContains || ( sline.find( "contains" ) != string::npos ) );

    //search for words indicating, that we did not reach a point where
    //we can insert variable definitions, these keywords are:
    //program, function, result, subroutine, save, implicit, parameter,
    //and use

    bool func = ( ( lline.find( "function" )   != string::npos )           &&
                  !( lline.find( "function" ) > lline.find( "pure" ) )          &&
                  !( lline.find( "function" ) > lline.find( "elemental" ) )     &&
                  !( lline.length() >= ( lline.find( "function" ) + 8 )        &&
                     ( isalnum( lline[ lline.find( "function" ) + 8 ] ) ||
                       ( lline[ lline.find( "function" ) + 8 ] == '_' ) ) )     &&
                  !( ( lline.find( "function" ) != 0 )                       &&
                     ( isalnum( lline[ lline.find( "function" ) - 1 ] )      ||
                       ( lline[ lline.find( "function" ) - 1 ] == '_' ) ) )  &&
                  ( isalpha( sline[ sline.find( "function" ) + 8 ] ) ||
                    sline[ sline.find( "function" ) + 8 ] == '&' ) );

    bool sub = ( ( lline.find( "subroutine" )   != string::npos )                 &&
                 !( lline.find( "subroutine" ) > lline.find( "pure" ) )          &&
                 !( lline.find( "subroutine" ) > lline.find( "elemental" ) )     &&
                 !( lline.length() >= ( lline.find( "subroutine" ) + 10 )     &&
                    ( isalnum( lline[ lline.find( "subroutine" ) + 10 ] ) ||
                      ( lline[ lline.find( "subroutine" ) + 10 ] == '_' ) ) )  &&
                 !( ( lline.find( "subroutine" ) != 0 )                     &&
                    ( isalnum( lline[ lline.find( "subroutine" ) - 1 ] )   ||
                      ( lline[ lline.find( "subroutine" ) - 1 ] == '_' ) ) ) &&
                 ( isalpha( sline[ sline.find( "subroutine" ) + 10 ] ) ||
                   sline[ sline.find( "subroutine" ) + 10 ] == '&' ) );

    bool key = ( ( sline.find( "save" )     == 0 && m_in_header ) ||
                 ( sline.find( "result" )   == 0 && m_in_header ) ||
                 ( sline.find( "implicit" ) == 0 && m_in_header ) ||
                 ( sline.find( "use" )      == 0 && m_in_header ) ||
                 ( sline.find( "include" )  == 0 && m_in_header ) );

    bool noend = ( sline.find( "endfunction" )   == string::npos &&
                   sline.find( "endsubroutine" ) == string::npos &&
                   sline.find( "endmodule" )     == string::npos &&
                   sline.find( "endprogram" )    == string::npos );

    size_t pos_e      = sline.find( "=" );
    bool   validequal = ( pos_e >= sline.find( "!" )          ||
                          pos_e > sline.find( "kind" )        ||
                          pos_e > sline.find( "only:" )       ||
                          pos_e == ( sline.find( ">" ) - 1 )  ||
                          pos_e > sline.find( "bind(" ) );

    bool misc = ( ( sline.find( "#" ) == 0 && m_in_header )         ||
                  ( sline.find( "&" ) == 0 && m_in_header )         ||
                  ( sline.empty()  && m_in_header )                 ||
                  ( sline.find( "parameter" ) == 0 && m_in_header ) ||
                  ( sline.find( "dimension" ) == 0 && m_in_header ) ||
                  ( openbrackets != 0 && m_in_header )              ||
                  ( continuation && m_in_header ) );

    /* Debug output */
    /*std::cout << std::endl << lline << std::endl;
       std::cout << "continuation= " << continuation << std::endl;
       std::cout << "m_in_header= " << m_in_header << std::endl;
       std::cout << "func= " << func << std::endl;
       std::cout << "sub= " << sub << std::endl;
       std::cout << "key= " << key << std::endl;
       std::cout << "validequal= " << validequal << std::endl;
       std::cout << "misc= " << misc << std::endl;
       std::cout << "openbrackets= " << openbrackets << std::endl;
       std::cout << "inModule= " << inModule << std::endl;
       std::cout << "inInterface= " << inInterface << std::endl;
       std::cout << "inContains= " << inContains << std::endl;
       std::cout << "noend= " << noend << std::endl;*/

    if ( ( ( sline.find( "program" ) == 0 || func || sub || key ) &&
           noend && validequal )     ||
         misc )
    {
        result = !inModule || ( !inInterface && inContains );
    }
    else
    {
        result = false;
    }
    //Check if we leave a program block
    inProgram = inProgram && sline.find( "endprogram" ) == string::npos;
    //Check if we leave a module block
    inModule = inModule && sline.find( "endmodule" ) == string::npos;
    //Check if we leave an interface block
    inInterface = inInterface && sline.find( "endinterface" ) == string::npos;
    //Check if we leave an contains block
    inContains = inContains && sline.find( "endmodule" ) == string::npos;

    if ( sline.length() && sline[ sline.length() - 1 ] == '&' )
    {
        continuation = true;
    }
    else
    {
        continuation = false;
    }

    /*count open brackets, to see if a functionheader is split across different lines*/
    for ( string::size_type i = 0; i < m_lowline.length(); i++ )
    {
        bool in_string = false;
        if ( m_lowline[ i ] == '(' )
        {
            openbrackets++;
        }
        if ( m_lowline[ i ] == ')' )
        {
            openbrackets--;
        }
        if ( ( m_lowline[ i ] == '\'' || m_lowline[ i ] == '"' ) && in_string )
        {
            in_string = false;
        }
        else
        {
            in_string = true;
        }
        /*rest of line is a comment*/
        if ( m_lowline[ i ] == '!' && !in_string )
        {
            break;
        }
    }

    return result;
}

bool
OPARI2_FortranParser::is_empty_line( void )
{
    return m_lowline.find_first_not_of( " \t" ) == string::npos;
}

/**@brief check if this line is a comment line*/
bool
OPARI2_FortranParser::is_comment_line( void )
{
    if ( m_options.form & F_FIX )
    {
        if ( m_lowline[ 0 ] == '!' ||
             m_lowline[ 0 ] == '*' ||
             m_lowline[ 0 ] == 'c' )
        {
            // fixed form comment
            if ( m_lowline[ 1 ] == '$' &&
                 m_lowline.find_first_not_of( " \t0123456789", 2 ) > 5 )
            {
                // Conditional Compilation for fixed form
                m_lowline[ 0 ] = ' ';
                m_lowline[ 1 ] = ' ';
                return false;
            }
            else if ( m_lowline[ 1 ] == 'p' && m_lowline[ 2 ] == '$' &&
                      m_lowline.find_first_not_of( " \t0123456789", 3 ) > 5 )
            {
                /**  POMP Conditional Compilation for fixed form. This
                 *  allows code to be only active after processing
                 *  with OPARI2 by using the !P$ sentinel to start a
                 *  comment */
                m_lowline[ 0 ] = m_line[ 0 ] = ' ';
                m_lowline[ 1 ] = m_line[ 1 ] = ' ';
                m_lowline[ 2 ] = m_line[ 2 ] = ' ';
                return false;
            }
            else
            {
                return true;
            }
        }
    }

    // free form comment
    size_t first_char = m_lowline.find_first_not_of( " \t" );
    if ( ( first_char != string::npos ) && ( m_lowline[ first_char ] == '!' ) )
    {
        if ( m_lowline[ first_char + 1 ] == '$' &&
             ( m_lowline[ first_char + 2 ] == ' ' || m_lowline[ first_char + 2 ] == '\t' ) )
        {
            // Conditional Compilation for free form
            m_lowline[ first_char ]     = ' ';
            m_lowline[ first_char + 1 ] = ' ';
            return false;
        }
        else if ( m_lowline[ first_char + 1 ] == 'p' && m_lowline[ first_char + 2 ] == '$' &&
                  ( m_lowline[ first_char + 3 ] == ' ' || m_lowline[ first_char + 3 ] == '\t' ) )
        {
            /**  POMP Conditional Compilation for fixed form. This
             *  allows code to be only active after processing with
             *  OPARI2 by using the !P$ sentinel to start a
             *  comment */
            m_lowline[ first_char ]     = m_line[ first_char ] = ' ';
            m_lowline[ first_char + 1 ] = m_line[ first_char + 1 ] = ' ';
            m_lowline[ first_char + 2 ] = m_line[ first_char + 2 ] = ' ';
            return false;
        }
        else
        {
            return true;
        }
    }

    return false;
}


/**@brief check if this line starts a do loop*/
bool
OPARI2_FortranParser::is_loop_start( string& label )
{
    string::size_type poslab = string::npos;

    label = "";
    if ( !( m_line.size() ) )
    {
        return false;
    }

    // is there a 'do '
    string::size_type pstart = m_lowline.find( "do" );
    if ( pstart == string::npos ||
         ( m_lowline[ pstart + 2 ] != '\0' &&
           m_lowline[ pstart + 2 ] != ' '  &&
           m_lowline[ pstart + 2 ] != '\t'    ) )
    {
        return false;
    }

    string::size_type pos = m_lowline.find_first_not_of( " \t" );
    // cerr << "pos: " << pos << std::endl;
    if ( pos != pstart )
    {
        // there is a DO_construct_name, i.e, this is a named do-loop
        poslab = m_lowline.find_first_of( ":", pos );
        if ( poslab == string::npos )
        {
            return false;
        }
        label = m_line.substr( pos, poslab - pos );
        // skip white space
        pos = m_lowline.find_first_not_of( " \t", poslab + 1 );
    }

    //check again, if pos now start of do, otherwise not a correct do statement
    pstart = m_lowline.find( "do", pos );
    if ( pstart != pos ||
         ( m_lowline[ pstart + 2 ] != '\0' &&
           m_lowline[ pstart + 2 ] != ' '  &&
           m_lowline[ pstart + 2 ] != '\t'    ) )
    {
        return false;
    }

    pos = m_lowline.find_first_not_of( " \t", pos + 2 );
    //     cerr << "pos2: " << pos << std::endl;
    if ( pos != string::npos && isdigit( m_lowline[ pos ] ) )
    {
        // there is a stmtlabel
        poslab = pos;
        pos    = m_lowline.find_first_not_of( "0123456789", pos );
        //         cerr << "2pos: " << pos << ", poslab: " << poslab << std::endl;
        label = m_line.substr( poslab, pos - poslab );
    }

    //    cerr << label << "\n\n";
    return true;
}

/**@brief check if this line is the end of a do loop*/
bool
OPARI2_FortranParser::is_loop_end( void )
{
    string toplabel = m_loop_stack.top().label;
    string label;

    if ( !( m_line.size() ) )
    {
        return false;
    }

    string::size_type pos = m_lowline.find_first_not_of( " \t" );

    // is it a nonblock DO loop?
    string::size_type poslab = toplabel.find_first_not_of( "0123456789" );
    if ( ( toplabel.size() > 0 ) && ( poslab == string::npos ) )
    {
        // search for nonblock Do loop
        poslab = pos;
        pos    = m_lowline.find_first_not_of( "0123456789", pos );

        // is there a label in this line?
        if ( poslab == pos )
        {
            return false;
        }
        label = m_line.substr( poslab, pos - poslab );

        // is it the label of the top loop
        if ( toplabel == label )
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    else
    {
        // search for block Do loop
        pos = m_lowline.find( "end", pos );
        if ( pos == string::npos || ( pos != m_lowline.find_first_not_of( " \t0123456789" ) ) )
        {
            return false;
        }

        //        pos = m_lowline.find( "do", pos + 3 );
        pos = m_lowline.find_first_not_of( " \t", pos + 3 );
        if ( pos == string::npos  || ( m_lowline.find( "do", pos ) > pos ) )
        {
            return false;
        }

        pos = m_lowline.find( "do", pos );
        if ( pos == string::npos  || ( m_lowline.find( "=", pos ) < m_lowline.find( "!", pos ) ) )
        {
            return false;
        }

        // search for label
        if ( toplabel.size() )
        {
            // skip white space
            poslab = m_lowline.find_first_not_of( " \t", pos + 2 );
            pos    = m_lowline.find_first_of( " \t", poslab );
            if ( poslab == pos )
            {
                return false;
            }
            label = m_line.substr( poslab, pos - poslab );

            // is it the label of the top loop
            if ( toplabel == label )
            {
                return true;
            }
            else
            {
                return false;
            }
        }
        else
        {
            return true;                // end do without label
        }
    }
}


void
OPARI2_FortranParser::test_and_insert_enddo( void )
{
    if ( !m_loop_directives.empty() && m_waitfor_loopend )
    {
        m_waitfor_loopend = false;

        if ( m_loop_directives.top()->NeedsEndLoopDirective() )
        {
            OPARI2_Directive* d_new = m_loop_directives.top()->EndLoopDirective( m_lineno_loopend );
            ProcessDirective( d_new, m_os );
            m_loop_directives.pop();
            delete d_new;
        }
    }
}

/** @brief Delete comments and strings before the lines are parsed
 *         to avoid finding keywords in comments or strings.*/
void
OPARI2_FortranParser::del_strings_and_comments( void )
{
    // zero out string constants and free form comments
    for ( string::size_type i = 0; i < m_lowline.size(); ++i )
    {
        if ( m_in_string )
        {
            // inside string
            if ( m_lowline[ i ] == m_in_string )
            {
                m_lowline[ i ] = '@';
                ++i;
                if ( i >= m_lowline.size() )
                {
                    // eol: no double string delimiter -> string ends
                    m_in_string = 0;
                    break;
                }
                if ( m_lowline[ i ] != m_in_string )
                {
                    // no double string delimiter -> string ends
                    m_in_string = 0;
                    continue;
                }
            }
            m_lowline[ i ] = '@';
        }

        else if ( m_lowline[ i ] == '!' )
        {
            /* -- zero out partial line F90 comments -- */
            for (; i < m_lowline.size(); ++i )
            {
                m_lowline[ i ] = ' ';
            }
            break;
        }

        else if ( m_lowline[ i ] == '\'' || m_lowline[ i ] == '\"' )
        {
            m_in_string    = m_lowline[ i ];
            m_lowline[ i ] = '@';
        }
    }
}


bool
OPARI2_FortranParser::is_directive( void )
{
    m_sentinel = "";

    string::size_type pos = string::npos;
    if ( m_options.form == F_FIX )
    {
        if ( m_lowline.size()
             && ( m_lowline[ 0 ] == '!' || m_lowline[ 0 ] == 'c' || m_lowline[ 0 ] == '*' ) )
        {
            pos = 0;
        }
    }
    else if ( m_options.form == F_FREE )
    {
        string::size_type pstart = m_lowline.find_first_not_of( " \t" );
        if ( pstart != string::npos && m_lowline.size() > pstart + 1
             && m_lowline[ pstart ] == '!' )
        {
            pos = pstart;
        }
    }

    if ( pos == string::npos )
    {
        return false;
    }

    return IsValidSentinel( m_lowline, pos, m_sentinel );
}



bool
OPARI2_FortranParser::get_next_line( void )
{
    if ( getline( m_is, m_line ) )
    {
        /* workaround for bogus getline implementations */
        while ( m_line.size() == 1 && m_line[ 0 ] == '\0' )
        {
            get_next_line();
        }

        /* remove extra \r from Windows source files */
        if ( m_line.size() && *( m_line.end() - 1 ) == '\r' )
        {
            m_line.erase( m_line.end() - 1 );
        }

        ++m_lineno;
        m_lowline = m_line;
        transform( m_line.begin(), m_line.end(), m_lowline.begin(), fo_tolower() );

        /*check for Fortran continuation lines*/
        if ( m_options.form == F_FREE )
        {
            m_continuation = m_next_is_continuation;
            string::size_type amp = m_line.find_last_not_of( " \t" );
            if ( amp != string::npos && m_line[ amp ] == '&' )
            {
                m_continuation = true;
            }
            else
            {
                m_continuation = false;
            }
        }
        return true;
    }

    return false;
}


void
OPARI2_FortranParser::handle_line_directive( const string::size_type lstart )
{
    /* line directive */
    string::size_type loc = m_line.find_first_not_of( " \t", lstart + 1 );
    if ( loc != string::npos && isdigit( m_line[ loc ] ) )
    {
        string::size_type nrstart, nrend;
        string            filename;
        nrstart  = m_line.find_first_not_of( " \t", lstart + 1 );
        nrend    = m_line.find_first_of( " \t", nrstart + 1 );
        m_lineno = atoi( m_line.substr( nrstart, nrend ).c_str() ) - 1;
        nrstart  = m_line.find_first_not_of( " \t\"", nrend + 1 );
        nrend    = m_line.find_first_of( " \t\"", nrstart + 1 );
        filename = m_line.substr( nrstart, nrend - nrstart );
        if (  nrstart != 0 )
        {
            if ( filename[ 0 ] == '/' )
            {
                /*absolute path*/
                m_curr_file = filename;
            }
            else
            {
                /*relative path*/
                string path( m_options.infile );
                path        = path.substr( 0, path.find_last_of( "/" ) );
                m_curr_file = path + "/" + filename;
            }
        }
        m_os << m_line << std::endl;
    }
    else if ( m_line.substr( lstart + 1, 5 ) == "line "
              &&
              isdigit( m_line[ m_line.find_first_not_of( " \t", lstart + 5 ) ] ) )
    {
        int    nrstart, nrend;
        string filename;
        nrstart  = m_line.find_first_not_of( " \t", lstart + 5 );
        nrend    = m_line.find_first_of( " \t", nrstart + 1 );
        m_lineno = atoi( m_line.substr( nrstart, nrend ).c_str() ) - 1;
        nrstart  = m_line.find_first_not_of( " \t\"", nrend + 1 );
        nrend    = m_line.find_first_of( " \t\"", nrstart + 1 );
        if ( nrstart != 0 )
        {
            filename = m_line.substr( nrstart, nrend - nrstart );
            if ( filename[ 0 ] == '/' )
            {
                /*absolute path*/
                m_curr_file = filename;
            }
            else
            {
                /*relative path*/
                string path( m_options.infile );
                path        = path.substr( 0, path.find_last_of( "/" ) );
                m_curr_file = path + "/" + filename;
            }
        }
        m_os << m_line << std::endl;
    }
    else
    {
        /*keep other C/C++ preprocessor directives like #if and #endif*/
        m_os << m_line << std::endl;
    }
}

void
OPARI2_FortranParser::handle_directive( void )
{
    if ( m_in_header == true )
    {
        m_in_header = false;

        if ( !InstrumentationDisabled( D_FULL ) )
        {
            m_os << "      include \'" << m_options.incfile_nopath << "\'" << std::endl;
        }
        if ( m_options.keep_src_info )
        {
            m_os << "#line " << m_lineno << " \"" << m_curr_file << "\"" << "\n";
        }
    }

    vector<string> lines;
    lines.push_back( m_lowline );

    bool     found_continuation_line;
    string   prev_sentinel = m_sentinel;
    unsigned ignored_lines = 0;
    do
    {
        found_continuation_line = false;

        if ( m_options.form == F_FREE )
        {
            string::size_type com = m_lowline.find( m_sentinel ) + m_sentinel.length();
            com = m_lowline.find( "!", com );
            if ( com != string::npos )
            {
                --com;
            }
            string::size_type amp = m_lowline.find_last_not_of( " \t", com );
            if ( m_lowline[ amp ] == '&' )
            {
                found_continuation_line = true;
            }
            else
            {
                /** There is no continuation line, so jump out
                 * of while loop */
                break;
            }
        }

        if ( get_next_line() )
        {
            m_unprocessed_line = true;
            ignored_lines++;

            prev_sentinel = m_sentinel;
            if ( is_directive() )
            {
                if ( m_options.form == F_FIX )
                {
                    string::size_type p = m_lowline.find( m_sentinel ) + m_sentinel.length();
                    if ( ( m_sentinel == prev_sentinel &&
                           ( m_lowline[ p ] != ' '    &&
                             m_lowline[ p ] != '\t'   &&
                             m_lowline[ p ] != '0' ) ) )
                    {
                        found_continuation_line = true;
                    }
                }

                if ( found_continuation_line )
                {
                    m_unprocessed_line = false;
                    ignored_lines--;
                    lines.push_back( m_lowline );
                }
            }
            else if ( is_comment_line() )
            {
                found_continuation_line = true;
                m_sentinel              = prev_sentinel;
            }
            else if ( found_continuation_line && !is_empty_line() )
            {
                cerr << m_curr_file << ":" << m_lineno - 1
                     << ": ERROR: missing continuation line\n";
                cleanup_and_exit();
            }
        }
        else
        {
            break;
        }
    }
    while ( found_continuation_line );

    vector<string>    directive_prefix( 1, prev_sentinel );
    OPARI2_Directive* d_new =
        NewDirective( lines, directive_prefix,  m_options.lang, m_curr_file,
                      m_lineno - lines.size() - ignored_lines + 1 );

    if ( d_new )
    {
        m_waitfor_loopstart = d_new->NeedsEndLoopDirective();

        if ( !d_new->EndsLoopDirective() )
        {
            test_and_insert_enddo();
        }

        ProcessDirective( d_new, m_os );

        if ( d_new->NeedsEndLoopDirective() )
        {
            m_loop_directives.push( d_new );
        }
        else if ( d_new->EndsLoopDirective() )
        {
            m_loop_directives.pop();
            m_waitfor_loopend = false;
        }
        d_new         = NULL;
        m_need_pragma = false;
    }
    else
    {
        // Print orig code if directive is ignored.
        for ( vector<string>::const_iterator line = lines.begin();
              line != lines.end(); ++line )
        {
            m_os << *line << std::endl;
        }
    }
}


bool
OPARI2_FortranParser::is_free_offload_directive()
{
    string::size_type lstart = string::npos;

    return m_lowline.size()
           && ( ( lstart = m_lowline.find_first_not_of( " \t" ) ) != string::npos )
           && ( lstart == m_lowline.find( "!dir$" ) )
           && ( m_lowline.find( "offload" ) != string::npos );
}

void
OPARI2_FortranParser::handle_free_offload_directive()
{
    string::size_type pstart = 0;

    if ( ( ( pstart = m_lowline.find_first_not_of( " \t", pstart + 5 ) ) != string::npos ) &&
         ( ( m_lowline.find( "offload", pstart ) == pstart ) ||
           ( m_lowline.find( "omp", pstart ) == pstart &&
             ( ( pstart = m_lowline.find_first_not_of( " \t", pstart + 3 ) ) != string::npos ) &&
             m_lowline.find( "offload", pstart ) == pstart ) ) )
    {
        if ( ( ( pstart = m_lowline.find_first_not_of( " \t", pstart + 7 ) ) != string::npos ) &&
             m_lowline.find( "begin", pstart ) == pstart )
        {
            /*begin of an offload regions*/
            m_os << m_line << std::endl;
            DisableInstrumentation( D_FULL );
        }
        else
        {
            /*next pragma must be offloaded*/
            m_os << m_line << std::endl;
            m_offload_pragma = true;
        }
    }
    else if ( m_lowline.find( "end", pstart ) == pstart &&
              ( ( pstart = m_lowline.find_first_not_of( " \t", pstart + 3 ) ) != string::npos ) &&
              m_lowline.find( "offload", pstart ) == pstart )
    {
        /*end of an offload regions*/
        m_os << m_line << std::endl;
        EnableInstrumentation( D_FULL );
    }
    else if ( m_lowline.find( "attributes", pstart ) == pstart &&
              ( ( pstart = m_lowline.find_first_not_of( " \t", pstart + 10 ) ) != string::npos ) &&
              m_lowline.find( "offload", pstart ) == pstart )
    {
        string::size_type nstart = m_lowline.find( "::", pstart );
        nstart = m_lowline.find_first_not_of( " \t", nstart + 2 );
        string::size_type nend = m_lowline.find_first_of( " \t\n\0", nstart );
        m_offload_attribute = m_lowline.substr( nstart, nend - nstart );
        m_os << m_line << std::endl;
    }
    else
    {
        /*print lines with !dir$ where no offload is present and continue*/
        m_os << m_line << std::endl;
    }
}
void
OPARI2_FortranParser::handle_offloaded_functions( void )
{
    string::size_type pos_function          = m_lowline.find( "function" );
    string::size_type pos_subroutine        = m_lowline.find( "subroutine" );
    string::size_type pos_end               = m_lowline.find( "end" );
    string::size_type pos_offload_attribute = m_lowline.find( m_offload_attribute );
    string::size_type pos_offload_function  = m_lowline.find( m_current_offload_function );

    if ( ( ( pos_function   < pos_offload_attribute   ||
             pos_subroutine < pos_offload_attribute )    &&
           pos_offload_attribute != string::npos )          &&
         !( pos_end < pos_function ||
            pos_end < pos_subroutine ) )
    {
        /* begin of offload function or subroutine */
        DisableInstrumentation( D_FULL );
        m_current_offload_function = m_offload_attribute;
        m_offload_attribute        = "";
        m_offload_function         = true;
    }
    else if ( ( ( ( pos_function < pos_offload_function   ||
                    m_offload_function )                     &&
                  pos_end < pos_function )                      ||
                ( ( pos_subroutine < pos_offload_function ||
                    m_offload_subroutine )                   &&
                  pos_end < pos_subroutine ) )                  &&
              pos_offload_function != string::npos )
    {
        /*end of offload function*/
        EnableInstrumentation( D_FULL );
        m_offload_function         = false;
        m_current_offload_function = "";
    }
}

void
OPARI2_FortranParser::handle_normal_line()
{
    bool isComment = is_comment_line();
    if ( m_need_pragma && !isComment && m_lowline.find_first_not_of( " \t" ) != string::npos )
    {
        cerr << m_curr_file << ":" << m_lineno - 1
             << ": ERROR: missing continuation line\n";
        cleanup_and_exit();
    }
    else if ( !m_loop_directives.empty() && !isComment && m_lowline.find_first_not_of( " \t" ) != string::npos )
    {
        test_and_insert_enddo();
    }

    if ( isComment )
    {
        m_os << m_line << '\n';
    }
    else if ( ( m_line.size() == 0
                || m_lowline.find_first_not_of( " \t" ) == string::npos )
              && m_lineno != 0 )
    {
        // empty line
        m_os << m_line << '\n';
        ++m_lineno_loopend;
    }
    else
    {
        // really normal line
        /*if no pragma follows the offload, a function is offloaded and
         * the next pragma must be processed normally*/
        m_offload_pragma = false;
        del_strings_and_comments();
        /* split line at ; */
        string complete_line( m_line );
        string complete_m_lowline( m_lowline );
        size_t position = 0;
        while ( position != string::npos )
        {
            if ( position == 0 )
            {
                m_line =
                    complete_line.substr( position,
                                          min( complete_m_lowline.find( ";", position + 1 ),
                                               complete_m_lowline.size() + 1 ) - position );
                m_lowline =
                    complete_m_lowline.substr( position,
                                               min( complete_m_lowline.find( ";", position + 1 ),
                                                    complete_m_lowline.size() + 1 ) - position );
            }
            else
            {
                m_line =
                    complete_line.substr( position + 1,
                                          min( complete_m_lowline.find( ";", position + 1 ),
                                               complete_m_lowline.size() + 1 ) - position - 1 );
                m_lowline =
                    complete_m_lowline.substr( position + 1,
                                               min( complete_m_lowline.find( ";", position + 1 ),
                                                    complete_m_lowline.size() + 1 ) - position - 1 );
            }
            position = complete_m_lowline.find( ";", position + 1 );
            if ( is_sub_unit_header() )
            {
                m_in_header = true;
            }
            else if ( m_in_header == true )
            {
                m_in_header = false;
                if ( !InstrumentationDisabled( D_FULL ) )
                {
                    m_os << "      include \'" << m_options.incfile_nopath << "\'" << std::endl;
                }
                if ( m_options.keep_src_info )
                {
                    m_os << "#line " << m_lineno << " \"" << m_curr_file << "\"" << "\n";
                }
            }
            // replace rumtime API call in this line if possible
            ReplaceRuntimeAPI( m_lowline, m_line, m_curr_file, L_FORTRAN );

            if ( m_lineno != 0 )
            {
                if ( position == string::npos )
                {
                    m_os << m_line << '\n';
                }
                else
                {
                    m_os << m_line << ';';
                }
            }

            /* * instrumentation of the end of regions which are
             * implicitly ended after a single line of code */
            if ( !m_continuation )
            {
                HandleSingleLineDirective( m_lineno, m_os );
            }
            else
            {
            }
            // search for loop start statement
            string label;
            if ( is_loop_start( label ) )
            {
                LoopDescriptionT loop;

                loop.is_directive_loop = m_waitfor_loopstart;
                loop.label             = label;
                m_loop_stack.push( loop );
            }
            // search for loop end statement
            else if ( ( !m_loop_stack.empty() ) &&
                      is_loop_end() )
            {
                LoopDescriptionT top_loop = m_loop_stack.top();
                m_waitfor_loopend = top_loop.is_directive_loop;
                m_loop_stack.pop();
                if ( !m_loop_stack.empty() )
                {
                    top_loop = m_loop_stack.top();
                }
                else
                {
                    top_loop.is_directive_loop = false;
                    top_loop.label             = "<none>";
                }

                // more than one loop ending on same statement (only numerical labels)
                while ( ( top_loop.label.find_first_of( "0123456789" ) != string::npos )
                        && ( is_loop_end() )
                        )
                {
                    m_waitfor_loopend = top_loop.is_directive_loop;
                    m_loop_stack.pop();
                    if ( !m_loop_stack.empty() )
                    {
                        top_loop = m_loop_stack.top();
                    }
                    else
                    {
                        top_loop.is_directive_loop = false;
                        top_loop.label             = "<none>";
                    }
                }
                m_lineno_loopend = m_lineno;
            }
            else
            {
                // normal line
            }

            m_waitfor_loopstart = false;
        }
    }
}


/** @brief This function processes fortran files and searches for a
 *         place to insert variables, pragmas/directives and the begin
 *         and end of do loops which are needed to ensure correct
 *         instrumentation of parallel do constructs.*/
void
OPARI2_FortranParser::process()
{
    while ( m_unprocessed_line || get_next_line() )
    {
        m_normal_line      = false;
        m_unprocessed_line = false;

        if ( !( m_offload_attribute.empty() && m_current_offload_function.empty() ) )
        {
            handle_offloaded_functions();
        }

        string::size_type lstart = m_line.find_first_not_of( " \t" );
        if ( m_in_string && !is_comment_line() )
        {
            del_strings_and_comments();

            ReplaceRuntimeAPI( m_lowline, m_line, m_curr_file, L_FORTRAN );

            m_os << m_line << '\n';
        }
        else if ( lstart != string::npos && m_line[ lstart ] == '#' )
        {
            handle_line_directive( lstart );
        }
        else if ( is_directive() )
        {
            handle_directive();
        }
        else if ( is_free_offload_directive() )
        {
            handle_free_offload_directive();
        }
        else
        {
            handle_normal_line();
        }
    }
}
