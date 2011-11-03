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
 *  @brief      Functions needed in fortran and C/C++ to process OpenMP 
 *              pragmas.*/ 

#include <config.h>
#include "ompragma.h"
#include <iostream>

/** brief Find the pragma name.*/
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
    else if ( name == "no" || name == "inst" )     /*INST*/
    {
        name += find_next_word();                  /*INST*/
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

/** @brief Detects whether a num_threads clause is present. The num_threads clause is
 *         deleted in the pragma string and stored in the num_threads member variable.*/
void
OMPragma::find_numthreads()
{
	int pos_num_threads, pos_open, pos_close;
	for (vector<string>::iterator it = lines.begin(); it != lines.end(); ++it)
	{
		pos_num_threads=it->find("num_threads");
		if (pos_num_threads != string::npos) 
		{
			pos_open=it->find("(",pos_num_threads);
			pos_close=it->find(")",pos_num_threads);
			num_threads = it->substr(pos_open+1 ,pos_close-pos_open-1);
			it->replace(pos_num_threads,pos_close-pos_num_threads+1,"");
			break;
		}
	}
}
