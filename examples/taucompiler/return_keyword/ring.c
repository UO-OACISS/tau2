#include <mpi.h>
#include <stdio.h>

#define SIZE 10
#define MacroFunctionReturn(a) { \
myTest(); \
return(a);\
}

#define SIZE 10
#define MacroVoidReturn { \
myTest2(); \
return;\
}

void myTest () {
  int me;
  MPI_Comm_rank (MPI_COMM_WORLD, &me);
  printf("%d return.\n", me);
}

void myTest2 () {
  int me;
  MPI_Comm_rank (MPI_COMM_WORLD, &me);
  printf("%d void return.\n", me);
}

int func(int me, int proc) {
  int i;
  int field[SIZE];
  MPI_Status status;

  for (i=0; i<SIZE; i++)
    field[i] = i;

  MPI_Barrier(MPI_COMM_WORLD);

  for (i=0; i<3; ++i) {
    if (me==0) {
      MPI_Send(&field, SIZE, MPI_INT, 1, 4711, MPI_COMM_WORLD);
      MPI_Recv(&field, SIZE, MPI_INT, proc-1, 4711, MPI_COMM_WORLD, &status);
    }
    else {
      MPI_Recv(&field, SIZE, MPI_INT, me-1, 4711, MPI_COMM_WORLD, &status);
      MPI_Send(&field, SIZE, MPI_INT, (me+1)%proc, 4711, MPI_COMM_WORLD);
    }
  }
  MPI_Bcast (&field, SIZE, MPI_INT, 0, MPI_COMM_WORLD);
  if (me>0) {
    MacroFunctionReturn(0);
  }
  printf("%d done.\n", me);
  MacroFunctionReturn(0);
}

void voidFunc(int me, int proc) {
  int i;
  int field[SIZE];
  MPI_Status status;

  for (i=0; i<SIZE; i++)
    field[i] = i;

  MPI_Barrier(MPI_COMM_WORLD);

  for (i=0; i<3; ++i) {
    if (me==0) {
      MPI_Send(&field, SIZE, MPI_INT, 1, 4711, MPI_COMM_WORLD);
      MPI_Recv(&field, SIZE, MPI_INT, proc-1, 4711, MPI_COMM_WORLD, &status);
    }
    else {
      MPI_Recv(&field, SIZE, MPI_INT, me-1, 4711, MPI_COMM_WORLD, &status);
      MPI_Send(&field, SIZE, MPI_INT, (me+1)%proc, 4711, MPI_COMM_WORLD);
    }
  }
  MPI_Bcast (&field, SIZE, MPI_INT, 0, MPI_COMM_WORLD);
  if (me>0) {
    MacroVoidReturn;
  }
  printf("%d void done.\n", me);
  MacroVoidReturn;
}

int main(int argc, char **argv) {
  int proc, me;

  MPI_Init (&argc, & argv);
  MPI_Comm_size (MPI_COMM_WORLD, &proc);
  MPI_Comm_rank (MPI_COMM_WORLD, &me);

  func(me, proc);
  voidFunc(me, proc);
    
  MPI_Finalize ();
}

