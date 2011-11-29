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
 *  @file       process_omp.cc
 *  @status     beta
 *
 *  @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
 *
 *  @authors    Bernd Mohr <b.mohr@fz-juelich.de>
 *              Dirk Schmidl <schmidl@rz-rwth-aachen.de>
 *              Peter Philippen <p.philippen@fz-juelich.de>
 *
 *  @brief      This file contains only the function process_pragma, which
 *              parses OpenMP pragmas and calls corresponding pragma handlers.*/

#include <config.h>
#include <iostream>
#ifdef EBUG
using std::cerr;

#  include <iomanip>
using std::setw;
# endif

#include "opari2.h"
#include "handler.h"

/** @brief Search for the pragma type and other info and call the
           matching handler*/
void
process_pragma( OMPragma* p,
                ostream&  os,
                bool*     hasEnd,
                bool*     isFor )
{
# ifdef EBUG
    for ( unsigned i = 0; i < p->lines.size(); ++i )
    {
        cerr << setw( 3 ) << p->lineno + i << ":O" << ( i ? "+" : " " )
             << ": " << p->lines[ i ] << "\n";
    }
# endif
    //search for the pragma name
    p->find_name();

    if ( do_transform || p->name == "instrument" )
    {
        if ( hasEnd )
        {
            *hasEnd = ( p->name != "barrier" && p->name != "noinstrument" &&
                        p->name != "flush"   && p->name != "threadprivate" &&
                        /*p->name != "ordered" && */ p->name != "taskwait" &&
#if defined( __GNUC__ ) && ( __GNUC__ < 3 )
                        p->name.substr( 0, 4 ) != "inst" );
#else
                        p->name.compare( 0, 4, "inst" ) != 0 );
#endif
        }
        if ( isFor  )
        {
            *isFor = ( p->name == "for" || p->name == "parallelfor" );
        }
        phandler_t handler = find_handler( p->name );
        handler( p, os );
    }
    else
    {
        for ( unsigned i = 0; i < p->lines.size(); ++i )
        {
            os << p->lines[ i ] << "\n";
        }
    }
    delete p;
}
