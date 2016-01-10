/****************************************************************************
 **                     TAU Portable Profiling Package                     **
 **                     http://www.cs.uoregon.edu/research/tau             **
 *****************************************************************************
 **    Copyright 1997-2009                                                 **
 **    Department of Computer and Information Science, University of Oregon **
 **    Advanced Computing Laboratory, Los Alamos National Laboratory        **
 ****************************************************************************/
/***************************************************************************
 **     File            : TauMpiT.c                                       **
 **     Description     : TAU Profiling Package                           **
 **     Contact         : tau-bugs@cs.uoregon.edu                         **
 **     Documentation   : See http://www.cs.uoregon.edu/research/tau      **
 ***************************************************************************/

//////////////////////////////////////////////////////////////////////
// Include Files 
//////////////////////////////////////////////////////////////////////
#include <mpi.h>
#include <Profile/Profiler.h> 
#include <Profile/TauEnv.h> 

//////////////////////////////////////////////////////////////////////
int Tau_mpi_t_initialize(void) {
  int returnVal, thread_provided;   

  returnVal = MPI_T_init_thread(MPI_THREAD_SINGLE, &thread_provided); 
  return returnVal; 
}

//////////////////////////////////////////////////////////////////////
void Tau_track_mpi_t(void) {

}

//////////////////////////////////////////////////////////////////////
void Tau_track_mpi_t_here(void) {

}

//////////////////////////////////////////////////////////////////////
void Tau_enable_tracking_mpi_t(void) {

}

//////////////////////////////////////////////////////////////////////
void Tau_disable_tracking_mpi_t(void) {

}


//////////////////////////////////////////////////////////////////////
// EOF : TauMpiT.c
//////////////////////////////////////////////////////////////////////


