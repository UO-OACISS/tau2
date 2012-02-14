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
 *  @file       process_c.cc
 *  @status     alpha
 *
 *  @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
 *
 *  @authors    Bernd Mohr <b.mohr@fz-juelich.de>
 *              Dirk Schmidl <schmidl@rz-rwth-aachen.de>
 *              Peter Philippen <p.philippen@fz-juelich.de>

 *  @brief     This file contains all functions to process C or C++ files.*/

#include <config.h>
#include <iostream>
using std::cerr;
#include <stack>
using std::stack;
#include <vector>
using std::vector;
#include <map>
using std::map;
#include <string>
using std::getline;
using std::string;
#include <cctype>
using std::isalnum;
using std::isalpha;

#include "opari2.h"
#include "handler.h"

namespace
{
string
find_next_word( vector<string>&    preStmt,
                unsigned           size,
                unsigned&          pline,
                string::size_type& ppos )
{
    while ( pline < size )
    {
        string::size_type wbeg = preStmt[ pline ].find_first_not_of( " \t", ppos );
        if ( preStmt[ pline ][ wbeg ] == '\\' || wbeg == string::npos )
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
            ppos = preStmt[ pline ].find_first_of( " \t()", wbeg );
            return preStmt[ pline ].substr( wbeg,
                                            ppos == string::npos ? ppos : ppos - wbeg );
        }
    }
    return "";
}

/** @brief preprocessor lines are passed to this function and checked, whether they are
 *         OpenMP pragmas.*/
bool
process_preStmt( vector<string>&   preStmt,
                 ostream&          os,
                 const char*       infile,
                 int               lineno,
                 string::size_type ppos,
                 bool*             e,
                 bool*             f,
                 bool              asd )
{
    unsigned       s         = preStmt.size();
    bool           inComment = false;

    vector<string> origStmt;

    // "remove" comments
    for ( unsigned i = 0; i < s; ++i )
    {
        string::size_type pos  = 0;
        string&           line = preStmt[ i ];
        string            origLine( line );

        origStmt.push_back( origLine );

        while ( pos < line.size() )
        {
            if ( inComment )
            {
                // look for comment end
                if ( line[ pos ] == '*' && line[ pos + 1 ] == '/' )
                {
                    line[ pos++ ] = ' ';
                    inComment     = false;
                }
                line[ pos++ ] = ' ';
            }
            else if ( line[ pos ] == '/' )
            {
                pos++;
                if ( line[ pos ] == '/' )
                {
                    // c++ comments
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
                    inComment       = true;
                }
            }
            else
            {
                pos++;
            }
        }
    }

    unsigned pline = 0;
    string   first = find_next_word( preStmt, s, pline, ppos );
    if ( first == "pragma" )
    {
        // OpenMP pragma?
        string word = find_next_word( preStmt, s, pline, ppos );
        if ( ( word == "omp" ) || ( word == "pomp" ) )
        {
            // check if the pragma contains a legal directive
            unsigned          pline2    = pline;
            string::size_type ppos2     = ppos;
            string            directive = find_next_word( preStmt, s, pline2, ppos2 );
            if ( ( directive == "inst" ) || ( directive == "instrument" ) ||
                 ( directive == "parallel" ) || ( directive == "threadprivate" ) ||
                 ( directive == "for" ) || ( directive == "flush" ) ||
                 ( directive == "barrier" ) || ( directive == "ordered" ) ||
                 ( directive == "sections" ) || ( directive == "section" ) ||
                 ( directive == "task" ) || ( directive == "taskwait" ) ||
                 ( directive == "master" ) || ( directive == "critical" ) ||
                 ( directive == "atomic" ) || ( directive == "single" ) ||
                 ( directive == "noinstrument" ) || ( directive == "noinst" ) )
            {
                OMPragmaC* p = new OMPragmaC( infile, lineno, pline, ppos, preStmt, asd );
                process_pragma( p, os, e, f );
                return true;
            }
        }
    }
    else if ( first == "include" )
    {
        // include <omp.h> -> remove it
        string word = find_next_word( preStmt, s, pline, ppos );
        if ( ( word == "\"omp.h\"" ) || ( word == "<omp.h>" ) )
        {
            s = 0;
        }
    }

    for ( unsigned i = 0; i < s; ++i )
    {
        os << origStmt[ i ] << "\n";
    }
    preStmt.clear();
    origStmt.clear();
    return false;
}
}

/** @brief Process C or C++ files and search for OpenMP pragmas and the related code blocks.
 * comments and strings are omited to avoid finding keywords in comments.*/
void
process_c_or_cxx( istream&    is,
                  const char* infile,
                  ostream&    os,
                  bool        addSharedDecl )
{
    string              line;
    bool                inComment   = false;
    bool                inString    = false;
    bool                preContLine = false;
    bool                requiresEnd = true;
    bool                isFor       = false;
    int                 lineno      = 1;
    string::size_type   pos         = 0;
    int                 level       = 0;
    int                 numSemi     = 0;
    string::size_type   lstart      = string::npos;
    vector<string>      preStmt;
    vector<string>      endStmt;
    stack<int>          nextEnd;
    map<string, string> wrapper;

    wrapper[ "omp_init_lock" ]         = "POMP2_Init_lock";
    wrapper[ "omp_destroy_lock" ]      = "POMP2_Destroy_lock";
    wrapper[ "omp_set_lock" ]          = "POMP2_Set_lock";
    wrapper[ "omp_unset_lock" ]        = "POMP2_Unset_lock";
    wrapper[ "omp_test_lock" ]         = "POMP2_Test_lock";
    wrapper[ "omp_init_nest_lock" ]    = "POMP2_Init_nest_lock";
    wrapper[ "omp_destroy_nest_lock" ] = "POMP2_Destroy_nest_lock";
    wrapper[ "omp_set_nest_lock" ]     = "POMP2_Set_nest_lock";
    wrapper[ "omp_unset_nest_lock" ]   = "POMP2_Unset_nest_lock";
    wrapper[ "omp_test_nest_lock" ]    = "POMP2_Test_nest_lock";

    nextEnd.push( -1 );

    while ( getline( is, line ) )
    {
        /* workaround for bogus getline implementations */
        if ( line.size() == 1 && line[ 0 ] == '\0' )
        {
            break;
        }

        /* remove extra \r from Windows source files */
        if ( line.size() && *( line.end() - 1 ) == '\r' )
        {
            line.erase( line.end() - 1 );
        }

        if ( preContLine )
        {
            /*
             * preprocessor directive continuation
             */
            preStmt.push_back( line );
            if ( line[ line.size() - 1 ] != '\\' )
            {
                preContLine = false;
                if ( process_preStmt( preStmt, os, infile, lineno - preStmt.size() + 1,
                                      lstart + 1, &requiresEnd, &isFor, addSharedDecl ) )
                {
                    if ( requiresEnd )
                    {
                        nextEnd.push( level );
                        numSemi = isFor ? 3 : 1;
                    }
                    else
                    {
                        numSemi = 0;
                    }
                }
            }
        }
        else if ( !inComment &&
                  ( ( lstart = line.find_first_not_of( " \t" ) ) != string::npos ) &&
                  line[ lstart ] == '#' )
        {
            /*
             * preprocessor directive
             */
            preStmt.push_back( line );
            if ( line[ line.size() - 1 ] == '\\' )
            {
                preContLine = true;
            }
            else
            {
                if ( process_preStmt( preStmt, os, infile, lineno, lstart + 1,
                                      &requiresEnd, &isFor, addSharedDecl ) )
                {
                    if ( requiresEnd )
                    {
                        nextEnd.push( level );
                        numSemi = isFor ? 3 : 1;
                    }
                    else
                    {
                        numSemi = 0;
                    }
                }
            }
        }
        else
        {
            /*
             * regular line
             */
            bool newlinePrinted = false;

            while ( pos < line.size() )
            {
                newlinePrinted = false;
                if ( inComment )
                {
                    // look for comment end
                    if ( line[ pos ] == '*' && line[ pos + 1 ] == '/' )
                    {
                        os << "*/";
                        inComment = false;
                        pos      += 2;
                    }
                    else
                    {
                        os << line[ pos++ ];
                    }
                }
                else if ( line[ pos ] == '/' )
                {
                    pos++;
                    if ( line[ pos ] == '/' )
                    {
                        // c++ comments
                        pos++;
                        os << "//";
                        while ( pos < line.size() )
                        {
                            os << line[ pos++ ];
                        }
                    }
                    else if ( line[ pos ] == '*' )
                    {
                        // c comment start
                        pos++;
                        os << "/*";
                        inComment = true;
                    }
                    else
                    {
                        os << '/';
                    }
                }
                else if ( inString || line[ pos ] == '\"' )
                {
                    // character string constant
                    if ( inString )
                    {
                        inString = false;
                        pos--; // to make sure current character gets reprocessed
                    }
                    else
                    {
                        os << "\"";
                    }
                    do
                    {
                        pos++;
                        if ( line[ pos ] == '\\' )
                        {
                            os << '\\';
                            pos++;
                            if ( line[ pos ] == '\0' )
                            {
                                inString = true;
                                break;
                            }
                            else if ( line[ pos ] == '\"' )
                            {
                                os << '\"';
                                pos++;
                            }
                            else if ( line[ pos ] == '\\' )
                            {
                                os << '\\';
                                pos++;
                                if ( line[ pos ] == '\0' )
                                {
                                    inString = true;
                                    break;
                                }
                            }
                        }
                        os << line[ pos ];
                    }
                    while ( line[ pos ] != '\"' );
                    pos++;
                }
                else if ( line[ pos ] == '\'' )
                {
                    // character constant
                    os << "\'";
                    do
                    {
                        pos++;
                        if ( line[ pos ] == '\\' )
                        {
                            os << '\\';
                            pos++;
                            if ( line[ pos ] == '\'' )
                            {
                                os << '\'';
                                pos++;
                            }
                        }
                        os << line[ pos ];
                    }
                    while ( line[ pos ] != '\'' );
                    pos++;
                }
                else if ( isalpha( line[ pos ] ) || line[ pos ] == '_' )
                {
                    // identifier
                    string::size_type startpos = pos;
                    while ( pos < line.size() &&
                            ( isalnum( line[ pos ] ) || line[ pos ] == '_' ) )
                    {
                        pos++;
                    }
                    string                        ident( line, startpos, pos - startpos );
                    map<string, string>::iterator w = wrapper.find( ident );
                    if ( w != wrapper.end() && instrument_locks() )
                    {
                        os << w->second;
                    }
                    else
                    {
                        os << ident;
                    }

                    if ( ident == "for" && numSemi == 1 )
                    {
                        numSemi = 3;
                    }
                }
                else if ( line[ pos ] == '{' )
                {
                    // block open
                    os << line[ pos++ ];
                    level++;
                    numSemi = 0;
                }
                else if ( line[ pos ] == '}' )
                {
                    // block close
                    os << line[ pos++ ];
                    level--;
                    if ( nextEnd.top() == level )
                    {
                        int moreChars = ( pos < line.size() );
                        os << '\n';
                        newlinePrinted = true;

                        // while because block can actually close more than one pragma
                        while ( nextEnd.top() == level )
                        {
                            // hack: use pline (arg3) for correction value for line info
                            process_pragma( new OMPragmaC( infile, lineno + 1 - moreChars,
                                                           1 - moreChars, 0, endStmt,
                                                           addSharedDecl ), os );
                            nextEnd.pop();
                        }
                        if ( moreChars )
                        {
                            for ( unsigned i = 0; i < pos; ++i )
                            {
                                os << ' ';
                            }
                        }
                    }
                }
                else if ( line[ pos ] == ';' )
                {
                    // statement end
                    os << line[ pos++ ];
                    numSemi--;
                    if ( numSemi == 0 )
                    {
                        int moreChars = ( pos < line.size() );
                        os << '\n';
                        newlinePrinted = true;
                        // hack: use pline (arg3) for correction value for line info
                        process_pragma( new OMPragmaC( infile, lineno + 1 - moreChars,
                                                       1 - moreChars, 0, endStmt,
                                                       addSharedDecl ), os );
                        nextEnd.pop();

                        // check whether statement actually closes more pragma
                        while ( nextEnd.top() == level )
                        {
                            // hack: use pline (arg3) for correction value for line info
                            process_pragma( new OMPragmaC( infile, lineno + 1 - moreChars,
                                                           1 - moreChars, 0, endStmt,
                                                           addSharedDecl ), os );
                            nextEnd.pop();
                        }
                        if ( moreChars )
                        {
                            for ( unsigned i = 0; i < pos; ++i )
                            {
                                os << ' ';
                            }
                        }
                    }
                }
                else
                {
                    os << line[ pos++ ];
                }
            }
            if ( !newlinePrinted )
            {
                os << '\n';
            }
        }
        ++lineno;
        pos = 0;
    }

    // check end position stack
    if ( nextEnd.top() != -1 )
    {
        cerr << "ERROR: could not determine end of OpenMP construct (braces mismatch?)\n";
        print_regstack_top();
        cleanup_and_exit();
    }
}
