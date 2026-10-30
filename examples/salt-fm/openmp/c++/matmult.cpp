#include <iostream>
#include <vector>

constexpr int N = 512;

using Matrix = std::vector<double>;

void initialize(Matrix &m) {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      m[i * N + j] = i + j;
    }
  }
}

void compute(const Matrix &a, const Matrix &b, Matrix &c) {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        c[i * N + j] += a[i * N + k] * b[k * N + j];
      }
    }
  }
}

void compute_interchange(const Matrix &a, const Matrix &b, Matrix &c) {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        c[i * N + j] += a[i * N + k] * b[k * N + j];
      }
    }
  }
}

int main() {
  Matrix a(N * N), b(N * N), c(N * N, 0.0);

  initialize(a);
  initialize(b);
  compute(a, b, c);
  compute_interchange(a, b, c);
  std::cout << "c[0][1] = " << c[1] << '\n';
  return 0;
}
