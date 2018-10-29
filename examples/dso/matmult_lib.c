void compute_lib(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  int i,j,k;
#pragma omp parallel private(i,j,k) shared(a,b,c)
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
#pragma omp for schedule(dynamic) nowait
    for (i=0; i<rows_a; i++) {
      for(j=0; j<cols_b; j++) {
        for (k=0; k<cols_a; k++) {
#ifdef APP_USE_INLINE_MULTIPLY
          c[i][j] += multiply(a[i][k], b[k][j]);
#else /* APP_USE_INLINE_MULTIPLY */
          c[i][j] += a[i][k] * b[k][j];
#endif /* APP_USE_INLINE_MULTIPLY */
        }
      }
    }
  }   /*** End of parallel region ***/
}

void compute_interchange_lib(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  int i,j,k;
#pragma omp parallel private(i,j,k) shared(a,b,c)
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
#pragma omp for schedule(dynamic) nowait
    for (i=0; i<rows_a; i++) {
      for (k=0; k<cols_a; k++) {
        for(j=0; j<cols_b; j++) {
#ifdef APP_USE_INLINE_MULTIPLY
          c[i][j] += multiply(a[i][k], b[k][j]);
#else /* APP_USE_INLINE_MULTIPLY */
          c[i][j] += a[i][k] * b[k][j];
#endif /* APP_USE_INLINE_MULTIPLY */
        }
      }
    }
  }   /*** End of parallel region ***/
}
