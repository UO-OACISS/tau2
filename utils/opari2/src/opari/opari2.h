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
#ifndef OPARI_H
#define OPARI_H

#include <iosfwd>
using std::istream;
using std::ostream;

#include "ompragma.h"
extern string pomp_tpd;
extern bool   copytpd;
extern bool   task_abort;
extern bool   task_warn;
extern bool   task_remove;
extern bool   untied_abort;
extern bool   untied_keep;
extern bool   untied_no_warn;

enum Language { L_NA  = 0x00,
                L_F77 = 0x01, L_F90 = 0x02, L_FORTRAN  = 0x03,
                L_C   = 0x04, L_CXX = 0x08, L_C_OR_CXX = 0x0C };

void
process_fortran( istream&    is,
                 const char* infile,
                 ostream&    os,
                 bool        addSharedDecl,
                 char*       incfile,
                 Language    lang );

void
process_c_or_cxx( istream&    is,
                  const char* infile,
                  ostream&    os,
                  bool        addSharedDecl );
void
process_pragma( OMPragma* p,
                ostream&  os,
                bool*     hasEnd = 0,
                bool*     isFor = 0 );

void
cleanup_and_exit();

#endif
