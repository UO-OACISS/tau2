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
 *  @file       pomp2_fwrapper_base.c
 *
 *
 *  @brief      Basic fortan wrappers calling the C versions.*/

#include <config.h>

#include <stdio.h>
#include <stdlib.h>

#include "pomp2_fwrapper_base.h"
#include <opari2/pomp2_lib.h>
#include <opari2/pomp2_user_lib.h>
#include "pomp2_fwrapper_def.h"

extern int pomp2_tracing;

/* *INDENT-OFF*  */
void FSUB(POMP2_Finalize)() {
  POMP2_Finalize();
}

void FSUB(POMP2_Init)() {
  POMP2_Init();
}

void FSUB(POMP2_Off)() {
  pomp2_tracing = 0;
}

void FSUB(POMP2_On)() {
  pomp2_tracing = 1;
}

void FSUB(POMP2_Begin)(POMP2_Region_handle* regionHandle,
		       char*                ctc_string) {
  POMP2_Begin(regionHandle, ctc_string);
}

void FSUB(POMP2_End)(POMP2_Region_handle* regionHandle) {
  POMP2_End(regionHandle);
}

/* *INDENT-OFF*  */
/*
   *----------------------------------------------------------------
 * Fortran  Wrapper for OpenMP API
 ******----------------------------------------------------------------
 */
/* *INDENT-OFF*  */
#if defined(__ICC) || defined(__ECC) || defined(_SX)
#define CALLFSUB(a) a
#else
#define CALLFSUB(a) FSUB(a)
#endif

void FSUB(POMP2_Init_lock)(omp_lock_t *s) {
  if ( pomp2_tracing ) {
    fprintf(stderr, "%3d: init lock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_init_lock)(s);
}

void FSUB(POMP2_Destroy_lock)(omp_lock_t *s) {
  if ( pomp2_tracing ) {
    fprintf(stderr, "%3d: destroy lock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_destroy_lock)(s);
}

void FSUB(POMP2_Set_lock)(omp_lock_t *s) {
  if ( pomp2_tracing ) {
    fprintf(stderr, "%3d: set lock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_set_lock)(s);
}

void FSUB(POMP2_Unset_lock)(omp_lock_t *s) {
  if ( pomp2_tracing ) {
    fprintf(stderr, "%3d: unset lock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_unset_lock)(s);
}

int  FSUB(POMP2_Test_lock)(omp_lock_t *s) {
  if ( pomp2_tracing ) {
    fprintf(stderr, "%3d: test lock\n", omp_get_thread_num());
  }
  return CALLFSUB(omp_test_lock)(s);
}

#ifndef __osf__
void FSUB(POMP2_Init_nest_lock)(omp_nest_lock_t *s) {
  if ( pomp2_tracing ) {
    fprintf(stderr, "%3d: init nestlock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_init_nest_lock)(s);
}

void FSUB(POMP2_Destroy_nest_lock)(omp_nest_lock_t *s) {
  if ( pomp2_tracing ) {
    fprintf(stderr, "%3d: destroy nestlock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_destroy_nest_lock)(s);
}

void FSUB(POMP2_Set_nest_lock)(omp_nest_lock_t *s) {
  if ( pomp2_tracing ) {
    fprintf(stderr, "%3d: set nestlock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_set_nest_lock)(s);
}

void FSUB(POMP2_Unset_nest_lock)(omp_nest_lock_t *s) {
  if ( pomp2_tracing ) {
    fprintf(stderr, "%3d: unset nestlock\n", omp_get_thread_num());
  }
  CALLFSUB(omp_unset_nest_lock)(s);
}

int  FSUB(POMP2_Test_nest_lock)(omp_nest_lock_t *s) {
  if ( pomp2_tracing ) {
    fprintf(stderr, "%3d: test nestlock\n", omp_get_thread_num());
  }
  return CALLFSUB(omp_test_nest_lock)(s);
}
#endif
