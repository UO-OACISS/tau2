/*
 * This file is part of the Score-P software (http://www.score-p.org)
 *
 * Copyright (c) 2009-2011,
 * RWTH Aachen University, Germany
 *
 * Copyright (c) 2009-2011,
 * Gesellschaft fuer numerische Simulation mbH Braunschweig, Germany
 *
 * Copyright (c) 2009-2011,
 * Technische Universitaet Dresden, Germany
 *
 * Copyright (c) 2009-2011,
 * University of Oregon, Eugene, USA
 *
 * Copyright (c) 2009-2011, 2013
 * Forschungszentrum Juelich GmbH, Germany
 *
 * Copyright (c) 2009-2011,
 * German Research School for Simulation Sciences GmbH, Juelich/Aachen, Germany
 *
 * Copyright (c) 2009-2011,
 * Technische Universitaet Muenchen, Germany
 *
 * This software may be modified and distributed under the terms of
 * a BSD-style license. See the COPYING file in the package base
 * directory for details.
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
