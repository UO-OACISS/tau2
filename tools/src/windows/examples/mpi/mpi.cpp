
#include "stdafx.h"

/*** For regular profiling: ***/
#define PROFILING_ON 1
#define TAU_WINDOWS
#define TAU_DOT_H_LESS_HEADERS
#include <TAU.h>

#include <stdio.h>
#include <mpi.h>

void foo() {
	TAU_PROFILE("foo", "", TAU_USER);
	Sleep(1000);
}
void bar() {
	TAU_PROFILE("bar", "", TAU_USER);
	Sleep(2000);
}


int main(int argc, char **argv) {
  char name[BUFSIZ];
  int length;
  int rank;
  int size;

  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(0);
  TAU_PROFILE("main", "", TAU_DEFAULT);

  MPI_Init(&argc, &argv);
  MPI_Get_processor_name(name, &length);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("mpi: [%d/%d] %s: hello world\n", rank+1,size, name);
  foo();
  bar();
  MPI_Finalize();
}