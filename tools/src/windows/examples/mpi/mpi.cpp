
#include "stdafx.h"

/*** For regular profiling: ***/
#define PROFILING_ON 1
#pragma comment(lib, "tau-profile-static.lib")

/*** For callpath profiling: ***/
//#define PROFILING_ON 1
//#define TAU_CALLPATH 1
//#pragma comment(lib, "tau-callpath-static.lib")

/*** For tracing: ***/
//#define TRACING_ON 1
//#pragma comment(lib, "tau-trace-static.lib")
 
#define CSIZE 10


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

void func(int me, int proc) {
  int i;
  int field[CSIZE];
  MPI_Status status;

  for (i=0; i<CSIZE; i++)
    field[i] = i;

  MPI_Barrier(MPI_COMM_WORLD);

  for (i=0; i<3; ++i) {
    if (me==0) {
      MPI_Send(&field, CSIZE, MPI_INT, 1, 4711, MPI_COMM_WORLD);
      MPI_Recv(&field, CSIZE, MPI_INT, proc-1, 4711, MPI_COMM_WORLD, &status);
    }
    else {
      MPI_Recv(&field, CSIZE, MPI_INT, me-1, 4711, MPI_COMM_WORLD, &status);
      MPI_Send(&field, CSIZE, MPI_INT, (me+1)%proc, 4711, MPI_COMM_WORLD);
    }
  }
  MPI_Bcast (&field, CSIZE, MPI_INT, 0, MPI_COMM_WORLD);
  printf("%d done.\n", me);
}





int main(int argc, char **argv) {
  char name[BUFSIZ];
  int length;
  int rank;
  int size;

  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE("main", "", TAU_DEFAULT);

  MPI_Init(&argc, &argv);
  MPI_Get_processor_name(name, &length);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  printf("mpi: [%d/%d] %s: hello world\n", rank+1,size, name);
  foo();
  bar();
  func(rank, size);
  MPI_Finalize();
}
