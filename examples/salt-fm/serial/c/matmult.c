#include <stdio.h>
#include <stdlib.h>

#define N 512

void initialize(double m[N][N]) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      m[i][j] = i + j;
    }
  }
}

void compute(double a[N][N], double b[N][N], double c[N][N]) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

void compute_interchange(double a[N][N], double b[N][N], double c[N][N]) {
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}

int main(void) {
  double (*a)[N] = malloc(sizeof(double[N][N]));
  double (*b)[N] = malloc(sizeof(double[N][N]));
  double (*c)[N] = calloc(N, sizeof(double[N]));

  initialize(a);
  initialize(b);
  compute(a, b, c);
  compute_interchange(a, b, c);
  printf("c[0][1] = %g\n", c[0][1]);

  free(a);
  free(b);
  free(c);
  return 0;
}
