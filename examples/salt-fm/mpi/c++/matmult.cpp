#include <iostream>
#include <vector>
#include <mpi.h>

constexpr int N = 512;

using Matrix = std::vector<double>;

void initialize(Matrix &m) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      m[i * N + j] = i + j;
    }
  }
}

void multiply_row(const std::vector<double> &row, const Matrix &b, std::vector<double> &answer) {
  for (int j = 0; j < N; j++) {
    answer[j] = 0.0;
  }
  for (int k = 0; k < N; k++) {
    for (int j = 0; j < N; j++) {
      answer[j] += row[k] * b[k * N + j];
    }
  }
}

int main(int argc, char *argv[]) {
  int rank, size;
  std::vector<double> row(N), answer(N);
  Matrix a(N * N), b(N * N), c(N * N);
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    std::cerr << "Run with 2 or more processes\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  std::cout << "Process " << rank << " of " << size << " is active\n";

  if (rank == 0) {
    initialize(a);
    initialize(b);
  }
  MPI_Bcast(b.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    int numsent = 0;
    for (int i = 1; i < size; i++) {
      if (numsent < N) {
        MPI_Send(&a[numsent * N], N, MPI_DOUBLE, i, numsent + 1, MPI_COMM_WORLD);
        numsent++;
      } else {
        MPI_Send(nullptr, 0, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
      }
    }
    for (int i = 0; i < N; i++) {
      MPI_Recv(answer.data(), N, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      std::copy(answer.begin(), answer.end(), c.begin() + (status.MPI_TAG - 1) * N);
      if (numsent < N) {
        MPI_Send(&a[numsent * N], N, MPI_DOUBLE, status.MPI_SOURCE, numsent + 1, MPI_COMM_WORLD);
        numsent++;
      } else {
        MPI_Send(nullptr, 0, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
      }
    }
    std::cout << "c[0][1] = " << c[1] << '\n';
  } else {
    MPI_Recv(row.data(), N, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    while (status.MPI_TAG != 0) {
      multiply_row(row, b, answer);
      MPI_Send(answer.data(), N, MPI_DOUBLE, 0, status.MPI_TAG, MPI_COMM_WORLD);
      MPI_Recv(row.data(), N, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    }
  }

  MPI_Finalize();
  return 0;
}
