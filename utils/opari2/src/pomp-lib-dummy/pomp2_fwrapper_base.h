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
 */
#include <omp.h>
#include "pomp2_fwrapper_def.h"

#ifndef POMP2_FWRAPPER_BASE_H
#define POMP2_FWRAPPER_BASE_H
extern void
FSUB( omp_init_lock )( omp_lock_t* s );
extern void
FSUB( omp_destroy_lock )( omp_lock_t* s );
extern void
FSUB( omp_set_lock )( omp_lock_t* s );
extern void
FSUB( omp_unset_lock )( omp_lock_t* s );
extern int
FSUB( omp_test_lock )( omp_lock_t* s );
extern void
FSUB( omp_init_nest_lock )( omp_nest_lock_t* s );
extern void
FSUB( omp_destroy_nest_lock )( omp_nest_lock_t* s );
extern void
FSUB( omp_set_nest_lock )( omp_nest_lock_t* s );
extern void
FSUB( omp_unset_nest_lock )( omp_nest_lock_t* s );
extern int
FSUB( omp_test_nest_lock )( omp_nest_lock_t* s );

#endif
