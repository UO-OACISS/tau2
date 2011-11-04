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
 *  @file       pomp2_fwrapper.c
 *  @status     alpha
 *
 *  @maintainer Dirk Schmidl <schmidl@rz.rwth-aachen.de>
 *
 *  @brief      This file contains fortran wrapper functions.*/

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pomp2_lib.h"
#include "pomp2_fwrapper_def.h"

/*
 * Fortran wrappers calling the C versions
 */
/* *INDENT-OFF*  */
extern "C" {
void FSUB(POMP2_Atomic_enter)(POMP2_Region_handle* regionHandle, char* ctc_string) {
   POMP2_Atomic_enter(regionHandle, ctc_string);
}

void FSUB(POMP2_Atomic_exit)(POMP2_Region_handle* regionHandle ) {
   POMP2_Atomic_exit(regionHandle );
}

void FSUB(POMP2_Implicit_barrier_enter)(POMP2_Region_handle* regionHandle ) {
   POMP2_Implicit_barrier_enter(regionHandle );
}

void FSUB(POMP2_Implicit_barrier_exit)(POMP2_Region_handle* regionHandle ) {
   POMP2_Implicit_barrier_exit(regionHandle );
}

void FSUB(POMP2_Barrier_enter)(POMP2_Region_handle* regionHandle, char* ctc_string) {
   POMP2_Barrier_enter(regionHandle, ctc_string);
}

void FSUB(POMP2_Barrier_exit)(POMP2_Region_handle* regionHandle ) {
   POMP2_Barrier_exit(regionHandle );
}

void FSUB(POMP2_Flush_enter)(POMP2_Region_handle* regionHandle, char* ctc_string ) {
  POMP2_Flush_enter(regionHandle, ctc_string );
}

void FSUB(POMP2_Flush_exit)(POMP2_Region_handle* regionHandle ) {
   POMP2_Flush_exit(regionHandle );
}

void FSUB(POMP2_Critical_begin)(POMP2_Region_handle* regionHandle ) {
   POMP2_Critical_begin(regionHandle );
}

void FSUB(POMP2_Critical_end)(POMP2_Region_handle* regionHandle ) {
   POMP2_Critical_end(regionHandle );
}

void FSUB(POMP2_Critical_enter)(POMP2_Region_handle* regionHandle, char* ctc_string) {
   POMP2_Critical_enter(regionHandle, ctc_string);
}

void FSUB(POMP2_Critical_exit)(POMP2_Region_handle* regionHandle ) {
   POMP2_Critical_exit(regionHandle );
}

void FSUB(POMP2_Do_enter)(POMP2_Region_handle* regionHandle, char* ctc_string) {
   POMP2_For_enter(regionHandle, ctc_string);
}

void FSUB(POMP2_Do_exit)(POMP2_Region_handle* regionHandle ) {
   POMP2_For_exit(regionHandle );
}

void FSUB(POMP2_Master_begin)(POMP2_Region_handle* regionHandle, char* ctc_string ) {
  POMP2_Master_begin(regionHandle, ctc_string );
}

void FSUB(POMP2_Master_end)(POMP2_Region_handle* regionHandle ) {
   POMP2_Master_end(regionHandle );
}

void FSUB(POMP2_Parallel_begin)(POMP2_Region_handle* regionHandle ) {
   POMP2_Parallel_begin(regionHandle );
}

void FSUB(POMP2_Parallel_end)(POMP2_Region_handle* regionHandle ) {
   POMP2_Parallel_end(regionHandle );
}

void FSUB(POMP2_Parallel_fork)(POMP2_Region_handle* regionHandle, int *num_threads, char* ctc_string) {
  POMP2_Parallel_fork(regionHandle, *num_threads, "dummy");
}

void FSUB(POMP2_Parallel_join)(POMP2_Region_handle* regionHandle ) {
   POMP2_Parallel_join(regionHandle );
}

void FSUB(POMP2_Section_begin)(POMP2_Region_handle* regionHandle, char* ctc_string ) {
  POMP2_Section_begin(regionHandle, ctc_string );
}

void FSUB(POMP2_Section_end)(POMP2_Region_handle* regionHandle ) {
   POMP2_Section_end(regionHandle );
}

void FSUB(POMP2_Sections_enter)(POMP2_Region_handle* regionHandle, char* ctc_string) {
   POMP2_Sections_enter(regionHandle, ctc_string);
}

void FSUB(POMP2_Sections_exit)(POMP2_Region_handle* regionHandle ) {
   POMP2_Sections_exit(regionHandle );
}

void FSUB(POMP2_Single_begin)(POMP2_Region_handle* regionHandle ) {
   POMP2_Single_begin(regionHandle );
}

void FSUB(POMP2_Single_end)(POMP2_Region_handle* regionHandle ) {
   POMP2_Single_end(regionHandle );
}

void FSUB(POMP2_Single_enter)(POMP2_Region_handle* regionHandle, char* ctc_string) {
   POMP2_Single_enter(regionHandle, ctc_string);
}

void FSUB(POMP2_Single_exit)(POMP2_Region_handle* regionHandle ) {
   POMP2_Single_exit(regionHandle );
}

void FSUB(POMP2_Workshare_enter)(POMP2_Region_handle* regionHandle, char* ctc_string) {
   POMP2_Workshare_enter(regionHandle, ctc_string);
}

void FSUB(POMP2_Workshare_exit)(POMP2_Region_handle* regionHandle ) {
   POMP2_Workshare_exit(regionHandle );
}

void FSUB(POMP2_Assign_handle)(POMP2_Region_handle* regionHandle, char* ctc_string, int ctc_string_len) {
  char *str;
  str=(char*) malloc((ctc_string_len+1)*sizeof(char));
  strncpy(str,ctc_string,ctc_string_len);
  str[ctc_string_len]='\0';
  POMP2_Assign_handle(regionHandle,str);
  free(str);
}
}
