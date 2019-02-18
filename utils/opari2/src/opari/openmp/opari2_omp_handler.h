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
 * Copyright (c) 2009-2013,
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
 *  @file       opari2_omp_handler.h
 *
 *  @brief
 */



#ifndef OPARI2_OMP_HANDLER_H
#define OPARI2_OMP_HANDLER_H

#include <iostream>
using std::ostream;


void
h_omp_parallel( OPARI2_Directive* d,
                ostream&          os );


void
h_end_omp_parallel( OPARI2_Directive* d,
                    ostream&          os );


void
h_omp_for( OPARI2_Directive* d,
           ostream&          os );

void
h_end_omp_for( OPARI2_Directive* d,
               ostream&          os );

void
h_omp_do( OPARI2_Directive* d,
          ostream&          os );


void
h_end_omp_do( OPARI2_Directive* d,
              ostream&          os );


void
h_omp_sections( OPARI2_Directive* d,
                ostream&          os );


void
h_omp_section( OPARI2_Directive* d,
               ostream&          os );

void
h_end_omp_section( OPARI2_Directive* d,
                   ostream&          os );

void
h_end_omp_sections( OPARI2_Directive* d,
                    ostream&          os );


void
h_omp_single( OPARI2_Directive* d,
              ostream&          os );


void
h_end_omp_single( OPARI2_Directive* d,
                  ostream&          os  );


void
h_omp_master( OPARI2_Directive* d,
              ostream&          os );


void
h_end_omp_master( OPARI2_Directive* d,
                  ostream&          os );


void
h_omp_critical( OPARI2_Directive* d,
                ostream&          os );


void
h_end_omp_critical( OPARI2_Directive* d,
                    ostream&          os );


void
h_omp_parallelfor( OPARI2_Directive* d,
                   ostream&          os );


void
h_end_omp_parallelfor( OPARI2_Directive* d,
                       ostream&          os );


void
h_omp_paralleldo( OPARI2_Directive* d,
                  ostream&          os );


void
h_end_omp_paralleldo( OPARI2_Directive* d,
                      ostream&          os );


void
h_omp_parallelsections( OPARI2_Directive* d,
                        ostream&          os );


void
h_end_omp_parallelsections( OPARI2_Directive* d,
                            ostream&          os );


void
h_omp_barrier( OPARI2_Directive* d,
               ostream&          os );


void
h_omp_flush( OPARI2_Directive* d,
             ostream&          os );


void
h_omp_atomic( OPARI2_Directive* d,
              ostream&          os );

void
h_end_omp_atomic( OPARI2_Directive* d,
                  ostream&          os );

void
h_omp_workshare( OPARI2_Directive* d,
                 ostream&          os );


void
h_end_omp_workshare( OPARI2_Directive* d,
                     ostream&          os );


void
h_omp_parallelworkshare( OPARI2_Directive* d,
                         ostream&          os );


void
h_end_omp_parallelworkshare( OPARI2_Directive* d,
                             ostream&          os );


void
h_omp_ordered( OPARI2_Directive* d,
               ostream&          os );


void
h_end_omp_ordered( OPARI2_Directive* d,
                   ostream&          os );


void
h_omp_task( OPARI2_Directive* d,
            ostream&          os );


void
h_end_omp_task( OPARI2_Directive* d,
                ostream&          os );


void
h_omp_taskwait( OPARI2_Directive* d,
                ostream&          os );

void
h_omp_threadprivate( OPARI2_Directive* d,
                     ostream&          os );

void
extra_openmp_atomic_handler( OPARI2_Directive* d,
                             const int         lineno,
                             ostream&          os );


void
finalize_handler( ostream& os );

#endif
