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
 *  @file       ompragma.cc
 *  @status     alpha
 *
 *  @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
 *
 *  @authors    Bernd Mohr <b.mohr@fz-juelich.de>
 *              Dirk Schmidl <schmidl@rz-rwth-aachen.de>
 *              Peter Philippen <p.philippen@fz-juelich.de>
 *
 *  @brief      Functions needed in fortran and C/C++ to process OpenMP
 *              pragmas.*/

#include <config.h>
#include "ompragma.h"
#include <iostream>

/** @brief Find the pragma name.*/
void
OMPragma::find_name()
{
    string w;

    if ( lines.empty() )
    {
        // automatically generated END pragma for C/C++
        name = "$END$";
        return;
    }
    name = find_next_word();
    if ( name == "parallel"  || name == "endparallel" )
    {
        w = find_next_word();
        if ( w == "do"  || w == "sections" ||
             w == "for" || w == "workshare" /*2.0*/ )
        {
            name += w;
        }
    }
    else if ( name == "end" )
    {
        w     = find_next_word();
        name += w;
        if ( w == "parallel" )
        {
            w = find_next_word();
            if ( w == "do"  || w == "sections" ||
                 w == "for" || w == "workshare" /*2.0*/ )
            {
                name += w;
            }
        }
    }
    else if ( name == "no" || name == "inst" )   /*INST*/
    {
        name += find_next_word();                /*INST*/
    }
}

/** @brief Removes all unnecessary commas. */
void
OMPragma::remove_commas()
{
    int bracket_counter = 0;

    for ( unsigned int line = 0; line < lines.size(); line++ )
    {
        for ( unsigned int c = 0; c < lines[ line ].length(); c++ )
        {
            if ( lines[ line ][ c ] == '(' )
            {
                bracket_counter++;
            }
            if ( lines[ line ][ c ] == ')' )
            {
                bracket_counter--;
            }
            if ( bracket_counter == 0 && lines[ line ][ c ] == ',' )
            {
                lines[ line ][ c ] = ' ';
            }
        }
    }
}

/* @brief True if a nowait is present*/
bool
OMPragma::is_nowait()
{
    unsigned          dummy  = 0;
    string::size_type dummy2 = 0;
    return find_word( "nowait", dummy, dummy2 );
}

/* @brief True if a copyprivate is present*/
bool
OMPragma::has_copypriv()
{
    unsigned          dummy  = 0;
    string::size_type dummy2 = 0;
    return find_word( "copyprivate", dummy, dummy2 );
}

string
OMPragma::find_sub_name()
{
    string cname = find_next_word();
    if ( cname == "(" )
    {
        return find_next_word();
    }
    return "";
}

/** @brief Detects whether a num_threads clause is present. The
 *         num_threads clause is deleted in the pragma string and
 *         stored in the arg_num_threads member variable.*/
bool
OMPragma::find_numthreads()
{
    unsigned          line = 0;
    string::size_type pos  = 0;

    if (  find_word( "num_threads", line, pos ) )
    {
        arg_num_threads = find_arguments( line, pos, true, "num_threads" );
        return true;
    }
    return false;
}

/** @brief Detects whether an if clause is present. The if
 *         clause is deleted in the pragma string and stored in the
 *         arg_if member variable.*/
bool
OMPragma::find_if()
{
    unsigned          line = 0;
    string::size_type pos  = 0;

    if (  find_word( "if", line, pos ) )
    {
        arg_if = find_arguments( line, pos, true, "if" );
        return true;
    }
    return false;
}

/** @brief Detects whether an if clause is present. The reduction
 *         clause is deleted in the pragma string and stored in the
 *         arg_reduction member variable.*/
bool
OMPragma::find_reduction()
{
    unsigned          line = 0;
    string::size_type pos  = 0;

    if (  find_word( "reduction", line, pos ) )
    {
        arg_reduction = find_arguments( line, pos, false, "reduction" );
        return true;
    }
    return false;
}

/** @brief Detects whether an schedule clause is present. The
 *         schedule clause is deleted in the pragma string and
 *         stored in the arg_schedule member variable.*/
bool
OMPragma::find_schedule( string* reg_arg_schedule )
{
    unsigned          line = 0;
    string::size_type pos  = 0;

    if ( find_word( "schedule", line, pos ) )
    {
        arg_schedule      = find_arguments( line, pos, false, "schedule" );
        *reg_arg_schedule = arg_schedule;
        return true;
    }
    return false;
}

/** @brief Detects whether an ordered clause is present.*/
bool
OMPragma::find_ordered()
{
    unsigned          line = 0;
    string::size_type pos  = 0;

    if (  find_word( "ordered", line, pos ) )
    {
        return true;
    }
    return false;
}

/** @brief Detects whether a collapse clause is present. The argument
 *         is stored in the arg_collapse member variable.*/
bool
OMPragma::find_collapse()
{
    unsigned          line = 0;
    string::size_type pos  = 0;

    if (  find_word( "collapse", line, pos ) )
    {
        arg_collapse = find_arguments( line, pos, false, "collapse" );
        return true;
    }
    return false;
}

/* @brief True if an untied is present*/
bool
OMPragma::find_untied( bool keep_untied )
{
    unsigned          line = 0;
    string::size_type pos  = 0;

    if (  find_word( "untied", line, pos ) )
    {
        if ( !keep_untied )
        {
            lines[ line ].replace( pos, 6, "      " );
        }
        return true;
    }
    return false;
}

/* @brief Is the default data sharing changed by default(none) or default(private) clause?*/
bool
OMPragma::changed_default()
{
    unsigned          line = 0;
    string::size_type pos  = 0;

    if (  find_word( "default(none)", line, pos ) || find_word( "default(private)", line, pos ) ||
          find_word( "default (none)", line, pos ) || find_word( "default (private)", line, pos ) )
    {
        return true;
    }
    return false;
}
