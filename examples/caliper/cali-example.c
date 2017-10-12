/************************************************************************************
 * **                      TAU Portable Profiling Package                          **
 * **                      http://www.cs.uoregon.edu/research/tau                  **
 * **********************************************************************************
 * **    Copyright 1997-2017                                                       **
 * **    Department of Computer and Information Science, University of Oregon      **
 * *********************************************************************************/
/***********************************************************************************
 * **      File            : cali-example.c                                        **
 * **      Description     : Example to demonstrate usage of TAU-CALIPER wrapper   **
 * **      Contact         : sramesh@cs.uoregon.edu                                **
 * **      Documentation   : See http://www.cs.uoregon.edu/research/tau            **
 * *********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef TAU_MPI
#include <mpi.h>
#endif /* TAU_MPI */

#include <omp.h>
#include <unistd.h>
#include <caliper/cali.h>


void init_dummy_function() {
  cali_begin_byname("init_dummy_function");
  sleep(1);
  cali_end_byname("init_dummy_function");

  cali_set_int_byname("randomval", 20);
}

int main(int argc, char **argv) {
  int i;
  char name[100] = "dummy";
  char stringified[10];
  /*Initialize CALIPER*/

#ifdef TAU_MPI
  MPI_Init(&argc, &argv);
#endif /* TAU_MPI */

  cali_init();

  cali_begin_int_byname("randomval", 10);

  /*Mark init*/
  cali_begin_byname("testing initialization");
  init_dummy_function();
  cali_end_byname("testing initialization");

#ifdef TAU_MPI
  MPI_Finalize();
#endif /* TAU_MPI */

  return 0;
}



