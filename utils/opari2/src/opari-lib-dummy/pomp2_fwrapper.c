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
 *
 *  @brief      This file contains fortran wrapper functions.*/

#include <config.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opari2/pomp2_lib.h>
#include <opari2/pomp2_user_lib.h>
#include "pomp2_fwrapper_def.h"

/*
 * Fortran wrappers calling the C versions
 */
/* *INDENT-OFF*  */
void FSUB(POMP2_Atomic_enter)(POMP2_Region_handle* regionHandle, char* ctc_string) {
   POMP2_Atomic_enter(regionHandle, ctc_string);
}

void FSUB(POMP2_Atomic_exit)(POMP2_Region_handle* regionHandle ) {
   POMP2_Atomic_exit(regionHandle );
}

void FSUB(POMP2_Implicit_barrier_enter)( POMP2_Region_handle* regionHandle,
                                         POMP2_Task_handle*   pomp2_old_task) {
  POMP2_Implicit_barrier_enter( regionHandle, pomp2_old_task );
}

void FSUB(POMP2_Implicit_barrier_exit)( POMP2_Region_handle* regionHandle,
                                        POMP2_Task_handle*   pomp2_old_task) {
  POMP2_Implicit_barrier_exit( regionHandle, *pomp2_old_task );
}

void FSUB(POMP2_Barrier_enter)( POMP2_Region_handle* regionHandle,
                                POMP2_Task_handle*   pomp2_old_task,
                                char*                ctc_string) {
  POMP2_Barrier_enter( regionHandle, pomp2_old_task, ctc_string);
}

void FSUB(POMP2_Barrier_exit)( POMP2_Region_handle* regionHandle,
                               POMP2_Task_handle*   pomp2_old_task ) {
  POMP2_Barrier_exit( regionHandle, *pomp2_old_task );
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

void FSUB(POMP2_Parallel_begin)( POMP2_Region_handle* regionHandle,
                                 POMP2_Task_handle*   newTask,
                                 char*                ctc_string ){
  POMP2_Parallel_begin(regionHandle);
}

void FSUB(POMP2_Parallel_end)(POMP2_Region_handle* regionHandle) {
   POMP2_Parallel_end(regionHandle);
}

void FSUB(POMP2_Master_begin)(POMP2_Region_handle* regionHandle, char* ctc_string ) {
  POMP2_Master_begin(regionHandle, ctc_string );
}

void FSUB(POMP2_Master_end)(POMP2_Region_handle* regionHandle ) {
   POMP2_Master_end(regionHandle );
}

void FSUB(POMP2_Parallel_fork)(POMP2_Region_handle* regionHandle,
                               int*                 if_clause,
                               int*                 num_threads,
                               POMP2_Task_handle*   pomp2_old_task,
                               char*                ctc_string) {
  POMP2_Parallel_fork(regionHandle, *if_clause, *num_threads, pomp2_old_task, "dummy");
}

void FSUB(POMP2_Parallel_join)(POMP2_Region_handle* regionHandle,
                               POMP2_Task_handle*   pomp2_old_task ) {
  POMP2_Parallel_join(regionHandle, *pomp2_old_task );
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

void FSUB(POMP2_Ordered_exit)(POMP2_Region_handle* regionHandle ) {
   POMP2_Ordered_exit(regionHandle );
}

void FSUB(POMP2_Ordered_begin)(POMP2_Region_handle* regionHandle ) {
   POMP2_Ordered_begin(regionHandle );
}

void FSUB(POMP2_Ordered_end)(POMP2_Region_handle* regionHandle ) {
   POMP2_Ordered_end(regionHandle );
}

void FSUB(POMP2_Ordered_enter)(POMP2_Region_handle* regionHandle, char* ctc_string) {
   POMP2_Ordered_enter(regionHandle, ctc_string);
}

void FSUB(POMP2_Single_exit)(POMP2_Region_handle* regionHandle ) {
   POMP2_Single_exit(regionHandle );
}

void FSUB(POMP2_Task_create_begin)(POMP2_Region_handle* regionHandle,
                                   POMP2_Task_handle*   pomp2_new_task,
                                   POMP2_Task_handle*   pomp2_old_task,
                                   int*                 pomp2_if,
                                   char*                ctc_string){
  POMP2_Task_create_begin(regionHandle, pomp2_new_task, pomp2_old_task, *pomp2_if, ctc_string);
}

void FSUB(POMP2_Task_create_end)(POMP2_Region_handle* regionHandle,
                                 POMP2_Task_handle*   pomp2_old_task ){
  POMP2_Task_create_end(regionHandle, *pomp2_old_task);
}

void FSUB(POMP2_Task_begin)( POMP2_Region_handle* regionHandle,
                             POMP2_Task_handle* pomp2_new_task ){
  POMP2_Task_begin(regionHandle, *pomp2_new_task);
}

void FSUB(POMP2_Task_end)(POMP2_Region_handle* regionHandle){
  POMP2_Task_end(regionHandle);
}

void FSUB(POMP2_Untied_task_create_begin)(POMP2_Region_handle* regionHandle,
                                          POMP2_Task_handle*   pomp2_new_task,
                                          POMP2_Task_handle*   pomp2_old_task,
                                          int*                 pomp2_if,
                                          char*                ctc_string){
  POMP2_Task_create_begin(regionHandle, pomp2_new_task, pomp2_old_task, *pomp2_if, ctc_string);
}

void FSUB(POMP2_Untied_task_create_end)(POMP2_Region_handle* regionHandle,
                                        POMP2_Task_handle*   pomp2_old_task ){
    POMP2_Task_create_end(regionHandle, *pomp2_old_task);
}

void FSUB(POMP2_Untied_task_begin)( POMP2_Region_handle* regionHandle,
                                    POMP2_Task_handle*   pomp2_new_task ){
  POMP2_Task_begin(regionHandle, *pomp2_new_task);
}

void FSUB(POMP2_Untied_task_end)(POMP2_Region_handle* regionHandle){
  POMP2_Task_end(regionHandle);
}

void FSUB(POMP2_Taskwait_begin)(POMP2_Region_handle* regionHandle,
                                POMP2_Task_handle*   pomp2_old_task,
                                char*                ctc_string ){
  POMP2_Taskwait_begin(regionHandle, pomp2_old_task, ctc_string );
}

void FSUB(POMP2_Taskwait_end)(POMP2_Region_handle* regionHandle,
                              POMP2_Task_handle*   pomp2_old_task ){
  POMP2_Taskwait_end(regionHandle, *pomp2_old_task);
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

void FSUB(POMP2_USER_Assign_handle)(POMP2_Region_handle* regionHandle, char* ctc_string, int ctc_string_len) {
  char *str;
  str=(char*) malloc((ctc_string_len+1)*sizeof(char));
  strncpy(str,ctc_string,ctc_string_len);
  str[ctc_string_len]='\0';
  POMP2_USER_Assign_handle(regionHandle,str);
  free(str);
}

/*
   *----------------------------------------------------------------
 * Wrapper for omp_get_max_threads used in instrumentation
 *
 * In Fortran a wrapper function
 * pomp2_get_max_threads() is used, since it is not possible to
 * ensure, that omp_get_max_threads is not used in the user
 * program. We would need to parse much more of the Fortran
 * Syntax to detect these cases.  The Wrapper function avoids
 * double definition of this function and avoids errors.
 *
 ******----------------------------------------------------------------
 */
int
FSUB(POMP2_Lib_get_max_threads)(void)
{
  return omp_get_max_threads();
}
