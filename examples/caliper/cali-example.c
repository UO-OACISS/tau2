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

#include <mpi.h>

#include <omp.h>
#include <unistd.h>
#include <caliper/cali.h>


void init_dummy_function() {
  cali_begin_byname("init_dummy_function");
  cali_end_byname("init_dummy_function");

  cali_set_int_byname("randomval", 20);
}

int main(int argc, char **argv) {
  int i;
  char name[100] = "dummy";
  char stringified[10];
  /*Initialize CALIPER*/

  MPI_Init(&argc, &argv);

  cali_init();

  cali_begin_int_byname("randomval", 10);

  /*Mark init*/
  cali_begin_byname("testing initialization");
  init_dummy_function();
  cali_end_byname("testing initialization");

  MPI_Finalize();

  return 0;
}



