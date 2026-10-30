#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define N 512

void initialize(double m[N][N]) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      m[i][j] = i + j;
    }
  }
}

void multiply_row(double row[N], double b[N][N], double answer[N]) {
  for (int j = 0; j < N; j++) {
    answer[j] = 0.0;
  }
  for (int k = 0; k < N; k++) {
    for (int j = 0; j < N; j++) {
      answer[j] += row[k] * b[k][j];
    }
  }
}

int main(int argc, char *argv[]) {
  int rank, size;
  double row[N], answer[N];
  double (*a)[N] = malloc(sizeof(double[N][N]));
  double (*b)[N] = malloc(sizeof(double[N][N]));
  double (*c)[N] = malloc(sizeof(double[N][N]));
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    fprintf(stderr, "Run with 2 or more processes\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  printf("Process %d of %d is active\n", rank, size);

  if (rank == 0) {
    initialize(a);
    initialize(b);
  }
  MPI_Bcast(b, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    int numsent = 0;
    for (int i = 1; i < size; i++) {
      if (numsent < N) {
        MPI_Send(a[numsent], N, MPI_DOUBLE, i, numsent + 1, MPI_COMM_WORLD);
        numsent++;
      } else {
        MPI_Send(NULL, 0, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
      }
    }
    for (int i = 0; i < N; i++) {
      MPI_Recv(answer, N, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      memcpy(c[status.MPI_TAG - 1], answer, sizeof(answer));
      if (numsent < N) {
        MPI_Send(a[numsent], N, MPI_DOUBLE, status.MPI_SOURCE, numsent + 1, MPI_COMM_WORLD);
        numsent++;
      } else {
        MPI_Send(NULL, 0, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
      }
    }
    printf("c[0][1] = %g\n", c[0][1]);
  } else {
    MPI_Recv(row, N, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    while (status.MPI_TAG != 0) {
      multiply_row(row, b, answer);
      MPI_Send(answer, N, MPI_DOUBLE, 0, status.MPI_TAG, MPI_COMM_WORLD);
      MPI_Recv(row, N, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
  }

  free(a);
  free(b);
  free(c);
  MPI_Finalize();
  return 0;
}
