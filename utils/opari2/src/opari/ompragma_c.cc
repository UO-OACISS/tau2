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
 *  @file       ompragma_c.cc
 *  @status     beta
 *
 *  @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
 *
 *  @authors    Bernd Mohr <b.mohr@fz-juelich.de>
 *              Dirk Schmidl <schmidl@rz-rwth-aachen.de>
 *              Peter Philippen <p.philippen@fz-juelich.de>
 *
 *  @brief      Functions needed to handle C specific parts of the OpenMP
 *              Pragma handling.*/


#include <config.h>
#include "ompragma.h"
#include <iostream>
//#include <iomanip>
#include <sstream>
#include "common.h"

#include "opari2.h"

string
OMPragmaC::find_next_word()
{
    while ( pline < lines.size() )
    {
        string::size_type wbeg = lines[ pline ].find_first_not_of( " \t", ppos );
        if ( lines[ pline ][ wbeg ] == '\\' || wbeg == string::npos )
        {
            ++pline;
            if ( pline < lines.size() )
            {
                ppos = 0;
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

/** @brief Returns true if word in line.*/
bool
OMPragmaC::find_word( const string       word,
                      unsigned&          line,
                      string::size_type& pos )
{
    for ( unsigned i = line; i < lines.size(); ++i )
    {
        string::size_type w = lines[ i ].find( word );
        while ( w != string::npos )
        {
            char b;
            char a;
            //word may start at position 0 of a continuation line
            if ( w == 0 )
            {
                b = ' ';
            }
            else
            {
                b = lines[ i ][ w - 1 ];
            }
            if ( lines[ i ].length() > w + word.length() )
            {
                a = lines[ i ][ w + word.length() ];
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
                    w = lines[ i ].find( word, w );
                }
            }
        } /*
             if ( w != string::npos )
             {
             line = i;
             pos  = w;
             return true;
             }*/
        pos = 0;
    }
    return false;
}

/** @brief Returns the arguments of a clause. */
string
OMPragmaC::find_arguments( unsigned&          line,
                           string::size_type& pos,
                           bool               remove,
                           string             clause )
{
    string arguments;
    int    bracket_counter = 0;

    for ( unsigned int i = 0; i < clause.length(); i++ )
    {
        if ( remove )
        {
            lines[ line ][ pos ] = ' ';
        }
        pos++;
    }
    pos = lines[ line ].find_first_not_of( " \t", pos );
    if ( lines[ line ][ pos ] == '(' )
    {
        bracket_counter++;
        if ( remove )
        {
            lines[ line ][ pos ] = ' ';
        }
        pos++;
    }
    else
    {
        std::cerr << filename << ":" << lineno << ": ERROR: Expecting argument for "
                  << clause << " clause\n" << std::endl;
        cleanup_and_exit();
    }
    while ( bracket_counter > 0 )
    {
        if ( lines[ line ][ pos ] == '(' )
        {
            bracket_counter++;
        }
        if ( lines[ line ][ pos ] == '\\' )
        {
            pos = lines[ line ].length();
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
                std::cerr << lines[ line - 1 ] << std::endl;
                std::cerr << filename << ":" << lineno << ": ERROR: Missing ) for "
                          << clause << " clause\n";
                cleanup_and_exit();
            }
            pos = 0;
        }
        if (  lines[ line ][ pos ] == ')' )
        {
            bracket_counter--;
        }
    }

    //remove last bracket if necessary
    if ( remove )
    {
        lines[ line ][ pos ] = ' ';
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

/** @brief add a nowait to a pragma */
void
OMPragmaC::add_nowait()
{
    int lastline = lines.size() - 1;
    lines[ lastline ].append( " nowait" );
}

/** @brief add region descriptors to shared variable  list*/
void
OMPragmaC::add_descr( int n )
{
    std::ostringstream os;
    if ( asd )
    {
        // Workaround for Suse 11.3 & iomanip -> intel 11.1 compiler bug
        std::stringstream ids;
        ids << n;

        os << " POMP2_DLIST_";
        for ( int i = ids.str().length(); i < 5; i++ )
        {
            os << '0';
        }
        os << ids.str();

        // This can be used again, as soon as the corresponding bug with
        //         Suse 11.3 and iomanip is fixed by intel.  os << "
        //         POMP2_DLIST_" << std::setw( 5 ) << std::setfill( '0'
        //         ) << n;
    }
    else
    {
        // not 100% right but best we can do if compiler doesn't allow
        // macro replacement on pragma statements
        os << " shared(" << region_id_prefix << n << ")";
    }
    int lastline = lines.size() - 1;
    lines[ lastline ].append( os.str() );
}

namespace
{
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
                string::size_type& pos )
{
    char* optr = &( outer->lines[ line ][ pos ] );
    char* iptr = &( inner->lines[ line ][ pos ] );
    while ( *optr != ')' )
    {
        while ( *optr == '\\' )
        {
            // skip to next line
            ++line;
            if ( line >= outer->lines.size() )
            {
                return;
            }
            pos  = 0;
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

/** @brief remove empty lines.*/
void
OMPragmaC::remove_empties()
{
    // remove lines without content
    vector<string>::iterator it = lines.begin();
    while ( it != lines.end() )
    {
        string::size_type l = it->find_first_not_of( " \t&" );
        if ( l == string::npos || ( *it )[ l ] == '\\' )
        {
            it = lines.erase( it );
        }
        else
        {
            ++it;
        }
    }

    // make sure last line is not a continuated line
    int               lastline = lines.size() - 1;
    string::size_type lastpos  = lines[ lastline ].size() - 1;
    if ( lines[ lastline ][ lastpos ] == '\\' )
    {
        lines[ lastline ][ lastpos ] = ' ';
    }
}

/** @brief Split combined parallel and worksharing constructs in two
 *         seperate pragmas to allow the insertion of POMP function
 *         calles between the parallel and the worksharing construct.
 *         clauses need to be matched to the corresponding pragma*/
OMPragma*
OMPragmaC::split_combined()
{
    remove_commas();
    // make empty copy with continuation characters
    vector<string> innerLines;
    for ( unsigned i = 0; i < lines.size(); ++i )
    {
        innerLines.push_back( string( lines[ i ].size(), ' ' ) );
        if ( i != lines.size() )
        {
            innerLines[ i ][ innerLines[ i ].size() - 1 ] = '\\';
        }
    }

    // copy sentinel
    unsigned          line = 0;
    string::size_type pos  = 0;
    pos                      = lines[ 0 ].find( "#" );
    innerLines[ 0 ][ pos++ ] = '#';

    find_word( "pragma", line, pos );
    sreplace( innerLines[ line ], "pragma", pos );
    pos += 6;

    find_word( "omp", line, pos );
    sreplace( innerLines[ line ], "omp", pos );
    pos += 3;

    OMPragmaC* inner = new OMPragmaC( filename, lineno, 0, 0, innerLines, asd );

    // fix pragma name
    line = pos = 0;
    if ( find_word( "for", line, pos ) )
    {
        sreplace( lines[ line ], "   ", pos );
        sreplace( inner->lines[ line ], "for", pos );
    }
    line = pos = 0;
    if ( find_word( "sections", line, pos ) )
    {
        sreplace( lines[ line ], "        ", pos );
        sreplace( inner->lines[ line ], "sections", pos );
    }

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
        fix_clause_arg( this, inner, line, pos );
    }
    line = pos = 0;
    while ( find_word( "schedule", line, pos ) )
    {
        sreplace( lines[ line ], "        ", pos );
        sreplace( inner->lines[ line ], "schedule", pos );
        pos += 8;
        fix_clause_arg( this, inner, line, pos );
    }
    line = pos = 0;
    while ( find_word( "collapse", line, pos ) )
    {
        sreplace( lines[ line ], "        ", pos );
        sreplace( inner->lines[ line ], "collapse", pos );
        pos += 8;
        fix_clause_arg( this, inner, line, pos );
    }

    // final cleanup
    remove_empties();
    inner->remove_empties();

    return inner;
}
