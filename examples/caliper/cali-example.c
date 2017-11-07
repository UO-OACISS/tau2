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

  cali_set_int_byname("randomint", 20);
  cali_id_t id = cali_find_attribute("randomint");
  printf("TAU: Attribute randomint has ID %d\n", id);
  cali_set_int(id, 55);

  cali_set_double_byname("randomdouble", 20.5);
  cali_id_t id_d = cali_find_attribute("randomdouble");
  printf("TAU: Attribute randomdouble has ID %d\n", id_d);
  cali_set_double(id_d, 55.0);

  //One should expect a warning here
  cali_begin_double_byname("randomdouble", 4.5);

  //Should not be a warning here
  cali_begin_double_byname("anotherrandomdouble", 4.5);

  //Dummy sleep
  sleep(1);
  cali_end_byname("init_dummy_function");

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



