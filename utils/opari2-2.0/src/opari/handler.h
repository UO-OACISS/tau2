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
#ifndef HANDLER_H
#define HANDLER_H

#include <sys/time.h>
#include "opari2.h"
#include "ompragma.h"

typedef void ( *phandler_t )( OMPragma*, ostream& );

void
init_handler( const char* infile,
              Language    l,
              bool        genLineStmts );

void
finalize_handler( const char* incfile,
                  ostream&    os );

phandler_t
find_handler( const string& pragma );

void
extra_handler( int      lineno,
               ostream& os );

bool
set_disabled( const string& constructs );

bool
instrument_locks();

bool
genLineStmts();

void
print_regstack_top();

extern bool    do_transform;
extern timeval compiletime;
#endif
