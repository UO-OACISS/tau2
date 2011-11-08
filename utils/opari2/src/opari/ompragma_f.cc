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
 *  @file       ompragma_f.cc
 *  @status     beta
 *
 *  @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
 *
 *  @authors    Bernd Mohr <b.mohr@fz-juelich.de>
 *              Dirk Schmidl <schmidl@rz-rwth-aachen.de>
 *              Peter Philippen <p.philippen@fz-juelich.de>
 *
 *  @brief      This file contains Fortran specific functions needed
 *              to handle OpenMP pragmas.*/

#include <config.h>
#include "ompragma.h"
#include <iostream>
/** @brief Find the next word in a line.*/
string
OMPragmaF::find_next_word()
{
    while ( pline < lines.size() )
    {
        string::size_type wbeg = lines[ pline ].find_first_not_of( " \t", ppos );
        if ( lines[ pline ][ wbeg ] == '&' || wbeg == string::npos )
        {
            ++pline;
            if ( pline < lines.size() )
            {
                ppos = lines[ pline ].find( sentinel ) + slen;
                if ( lines[ pline ][ ppos ] == '&' )
                {
                    ++ppos;
                }
            }
            else
            {
                return "";
            }
        }
        else if ( lines[ pline ][ wbeg ] == '(' || lines[ pline ][ wbeg ] == ')' )
        {
            ppos = wbeg + 1;
            return string( 1, lines[ pline ][ wbeg ] );
        }
        else
        {
            ppos = lines[ pline ].find_first_of( " \t()", wbeg );
            return lines[ pline ].substr( wbeg, ppos == string::npos ? ppos : ppos - wbeg );
        }
    }
    return "";
}

/** @brief True if word is in line.*/
bool
OMPragmaF::find_word( const char*        word,
                      unsigned&          line,
                      string::size_type& pos )
{
    for ( unsigned i = line; i < lines.size(); ++i )
    {
        string::size_type s = ( pos == 0 ) ? lines[ i ].find( sentinel ) + slen : pos;
        string::size_type w = lines[ i ].find( word, s );
        string::size_type c = lines[ i ].find( '!', s );
        // if word found and found before comment
        if ( w != string::npos &&
             ( c == string::npos || ( c != string::npos && w < c ) )
             )
        {
            line = i;
            pos  = w;
            return true;
        }
        pos = 0;
    }
    return false;
}

/** @brief Returns the arguments of a clause. */
string
OMPragmaF::find_arguments( unsigned&          line,
                           string::size_type& pos,
                           bool               remove )
{
    string arguments;
    bool   skip_first_amp = false;

    while ( lines[ line ][ pos ] != ')' )
    {
        if ( lines[ line ][ pos ] == '&' && !skip_first_amp )
        {
            pos            = lines[ line ].length();
            skip_first_amp = false;
        }
        else
        {
            arguments.append( 1, lines[ line ][ pos ] );
            if ( remove )
            {
                lines[ line ][ pos ] = ' ';
            }
            pos++;
        }
        if ( pos >= lines[ line ].length() )
        {
            line++;
            if ( line >= lines.size() )
            {
                std::cerr << "ERROR: Missing )\n";
                break;
            }
            else
            {
                pos = lines[ line ].find_first_of( "!*cC" ) + 6;
                int firstcharpos = lines[ line ].find_first_not_of( " \t", pos );
                if ( lines[ line ][ firstcharpos ] == '&' )
                {
                    skip_first_amp = true;
                }
            }
        }
    }

    return arguments;
}

/** @brief Add a nowait at the end of worksharing constructs, if needed.*/
void
OMPragmaF::add_nowait()
{
    int               lastline = lines.size() - 1;
    string::size_type s        = lines[ lastline ].find( sentinel ) + slen;
    // insert on last line on last position before comment
    string::size_type c = lines[ lastline ].find( '!', s );
    if ( c == string::npos )
    {
        lines[ lastline ].append( " nowait" );
    }
    else
    {
        lines[ lastline ].insert( c, " nowait" );
    }
}

void
OMPragmaF::add_descr( int )
{
    /* current implementation doesn't need anything special for fortran */
}

namespace
{
inline void
sreplace( string&       lhs,
          const string& rhs,
          int           from,
          int           to )
{
    for ( int i = from; i < to; ++i )
    {
        lhs[ i ] = rhs[ i ];
    }
}

inline void
sreplace( string&     lhs,
          const char* rhs,
          int         from )
{
    do
    {
        lhs[ from ] = *rhs;
        ++from;
        ++rhs;
    }
    while ( *rhs );
}

void
fix_clause_arg( OMPragma*          outer,
                OMPragma*          inner,
                unsigned&          line,
                string::size_type& pos,
                const string&      sentinel,
                int                slen )
{
    char* optr = &( outer->lines[ line ][ pos ] );
    char* iptr = &( inner->lines[ line ][ pos ] );
    while ( *optr != ')' )
    {
        while ( *optr == '!' || *optr == '&' ||
                pos >= outer->lines[ line ].size() )
        {
            // skip to next line
            ++line;
            if ( line >= outer->lines.size() )
            {
                return;
            }
            pos = outer->lines[ line ].find( sentinel ) + slen;
            pos = outer->lines[ line ].find_first_not_of( " \t", pos );
            if ( outer->lines[ line ][ pos ] == '&' )
            {
                ++pos;
            }
            optr = &( outer->lines[ line ][ pos ] );
            iptr = &( inner->lines[ line ][ pos ] );
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
}
/** @brief Remove empty lines.*/
void
OMPragmaF::remove_empties()
{
    // remove lines without content
    vector<string>::iterator it = lines.begin();
    while ( it != lines.end() )
    {
        string::size_type s = it->find( sentinel ) + slen;
        string::size_type l = it->find_first_not_of( " \t&", s );
        if ( l == string::npos || ( *it )[ l ] == '!' )
        {
            it = lines.erase( it );
        }
        else
        {
            ++it;
        }
    }

    // make sure 1st line is not a continuation line
    string::size_type s = lines[ 0 ].find( sentinel );
    if ( s == 1 )
    {
        lines[ 0 ][ slen ] = ' ';
    }
    else
    {
        string::size_type l = lines[ 0 ].find_first_not_of( " \t", s + slen );
        if ( lines[ 0 ][ l ] == '&' )
        {
            lines[ 0 ][ l ] = ' ';
        }
    }

    // make sure last line is not a continuated line
    int               lastline = lines.size() - 1;
    s = lines[ lastline ].find( sentinel ) + slen;
    string::size_type c = lines[ lastline ].find( '!', s );
    if ( c != string::npos )
    {
        --c;
    }
    string::size_type amp = lines[ lastline ].find_last_not_of( " \t", c );
    if ( lines[ lastline ][ amp ] == '&' )
    {
        lines[ lastline ][ amp ] = ' ';
    }
}

/** @brief Split combined parallel and worksharing pragmas in two
 *         seperate pragmas to allow POMP function calls in between.*/
OMPragma*
OMPragmaF::split_combined()
{
    OMPragmaF* inner = new OMPragmaF( filename, lineno, 0,
                                      string( lines[ 0 ].size(), ' ' ),
                                      ( slen == 6 ), asd );

    // copy sentinel and continuation characters
    for ( unsigned i = 0; i < lines.size(); ++i )
    {
        if ( i )
        {
            inner->lines.push_back( string( lines[ i ].size(), ' ' ) );
        }

        // sentinel (and column 6/7)
        string::size_type s = lines[ i ].find( sentinel );
        sreplace( inner->lines[ i ], lines[ i ], s - 1, s + slen );

        // & continuation characters
        string::size_type com = lines[ i ].find( "!", s + slen );
        if ( com != string::npos )
        {
            --com;
        }
        string::size_type amp2 = lines[ i ].find_last_not_of( " \t", com );
        if ( lines[ i ][ amp2 ] == '&' )
        {
            inner->lines[ i ][ amp2 ] = '&';
        }
        string::size_type amp1 = lines[ i ].find_first_not_of( " \t", s + slen );
        if ( lines[ i ][ amp1 ] == '&' )
        {
            inner->lines[ i ][ amp1 ] = '&';
        }
    }

    // fix pragma name
    unsigned          line = 0;
    string::size_type pos  = 0;
    if ( find_word( "do", line, pos ) )
    {
        sreplace( lines[ line ], "  ", pos );
        sreplace( inner->lines[ line ], "do", pos );
    }
    line = pos = 0;
    if ( find_word( "sections", line, pos ) )
    {
        sreplace( lines[ line ], "        ", pos );
        sreplace( inner->lines[ line ], "sections", pos );
    }
    line = pos = 0;                                         /*2.0*/
    if ( find_word( "workshare", line, pos ) )              /*2.0*/
    {
        sreplace( lines[ line ], "         ", pos );        /*2.0*/
        sreplace( inner->lines[ line ], "workshare", pos ); /*2.0*/
    }                                                       /*2.0*/

    // fix pragma clauses
    line = pos = 0;
    while ( find_word( "ordered", line, pos ) )
    {
        sreplace( lines[ line ], "       ", pos );
        sreplace( inner->lines[ line ], "ordered", pos );
        pos += 7;
    }
    line = pos = 0;
    while ( find_word( "lastprivate", line, pos ) )
    {
        sreplace( lines[ line ], "           ", pos );
        sreplace( inner->lines[ line ], "lastprivate", pos );
        pos += 11;
        fix_clause_arg( this, inner, line, pos, sentinel, slen );
    }
    line = pos = 0;
    while ( find_word( "schedule", line, pos ) )
    {
        sreplace( lines[ line ], "        ", pos );
        sreplace( inner->lines[ line ], "schedule", pos );
        pos += 8;
        fix_clause_arg( this, inner, line, pos, sentinel, slen );
    }
    line = pos = 0;
    while ( find_word( "collapse", line, pos ) )
    {
        sreplace( lines[ line ], "        ", pos );
        sreplace( inner->lines[ line ], "collapse", pos );
        pos += 8;
        fix_clause_arg( this, inner, line, pos, sentinel, slen );
    }

    // final cleanup
    remove_empties();
    inner->remove_empties();

    return inner;
}
