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
 *  @file       pomp2_fwrapper_base.c
 *  @status     alpha
 *
 *  @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
 *
 *  @brief      Basic fortan wrappers calling the C versions.*/  

#include <config.h>

#include <stdio.h>
#include <stdlib.h>

#include "pomp2_lib.h"
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

void FSUB(POMP2_Begin)(int* regionHandle) {
  if ( pomp2_tracing ) POMP2_Begin((void**)&regionHandle);
}

void FSUB(POMP2_End)(int* regionHandle) {
  if ( pomp2_tracing ) POMP2_End((void**)&regionHandle);
}
