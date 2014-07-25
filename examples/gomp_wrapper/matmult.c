/******************************************************************************
*   OpenMp Example - Matrix Multiply - C Version
*   Demonstrates a matrix multiply using OpenMP. 
*
*   Modified from here:
*   https://computing.llnl.gov/tutorials/openMP/samples/C/omp_mm.c
*
*   For  PAPI_FP_INS, the exclusive count for the event: 
*   for (null) [OpenMP location: file:matmult.c ]
*   should be  2E+06 / Number of Threads 
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <omp.h>
#include <math.h>

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 256
#endif

#define NRA MATRIX_SIZE                 /* number of rows in matrix A */
#define NCA MATRIX_SIZE                 /* number of columns in matrix A */
#define NCB MATRIX_SIZE                 /* number of columns in matrix B */

void initialize(double **matrix, int rows, int cols) {
  int i,j;
  printf("Initialize...\n");
  fflush(stdout);
#pragma omp parallel private(i,j) shared(matrix)
  {
    //set_num_threads();
    /*** Initialize matrices ***/
#pragma omp for nowait schedule(runtime)
    for (i=0; i<rows; i++) {
      for (j=0; j<cols; j++) {
        matrix[i][j]= i+j;
      }
    }
  }
}

double** allocateMatrix(int rows, int cols) {
  int i;
  printf("Allocate matrix...\n");
  fflush(stdout);
  double **matrix = (double**)malloc((sizeof(double*)) * rows);
  for (i=0; i<rows; i++) {
    matrix[i] = (double*)malloc((sizeof(double)) * cols);
  }
  return matrix;
}

#ifdef APP_USE_INLINE_MULTIPLY
__inline double multiply(double a, double b) {
	return a * b;
}
#endif /* APP_USE_INLINE_MULTIPLY */

// cols_a and rows_b are the same value
void compute(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  int i,j,k;
  printf("Compute...\n");
  fflush(stdout);
#pragma omp parallel private(i,j,k) shared(a,b,c)
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
#pragma omp for nowait schedule(runtime)
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

// cols_a and rows_b are the same value
void compute_triangular(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  int i,j,k;
  printf("Compute triangular...\n");
  fflush(stdout);
#pragma omp parallel private(i,j,k) shared(a,b,c)
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
#pragma omp for nowait schedule(runtime)
    for (i=0; i<rows_a; i++) {
      for(j=0; j<cols_b-i; j++) {
        for (k=0; k<cols_a-j; k++) {
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

void compute_interchange(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  int i,j,k;
  printf("Compute interchange...\n");
  fflush(stdout);
#pragma omp parallel private(i,j,k) shared(a,b,c)
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
#pragma omp for nowait schedule(runtime)
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

double do_work(void) {
  double **a,           /* matrix A to be multiplied */
  **b,           /* matrix B to be multiplied */
  **c;           /* result matrix C */
  printf("Do work...\n");
  fflush(stdout);
  a = allocateMatrix(NRA, NCA);
  b = allocateMatrix(NCA, NCB);
  c = allocateMatrix(NRA, NCB);  

/*** Spawn a parallel region explicitly scoping all variables ***/

  initialize(a, NRA, NCA);
  initialize(b, NCA, NCB);
  initialize(c, NRA, NCB);

  compute(a, b, c, NRA, NCA, NCB);
  compute_interchange(a, b, c, NRA, NCA, NCB);
  compute_triangular(a, b, c, NRA, NCA, NCB);

  return c[0][1]; 
}

int atomic () {
  int count = 0;
  int max = 2;
  #pragma omp parallel
  {
    #pragma omp atomic
    count++;
  }
  return count;
}

int busysleep(int tid) {
  int i;
  int dummy = 0;
  for (i = 0 ; i < tid*10000000 ; i++) {
    dummy += tid;
  }
  return dummy;
}

int barrier () {
  int count[1024] = {0};
  //int max = (omp_get_max_threads() < 2 ? 1 : omp_get_max_threads());
  int max = omp_get_max_threads();
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    printf("Thread %d sleeping %d seconds.\n", tid, tid);
    count[tid] = busysleep(tid);
    #pragma omp barrier
  }
  return count[max];
}

#define CRITICAL_SIZE 10

int critical() {
  int i;
  int max;
  int a[CRITICAL_SIZE];

  for (i = 0; i < CRITICAL_SIZE; i++) {
    a[i] = rand();
  }

  max = a[0];
  //int maxthreads = omp_get_max_threads() == 1 ? 1 : 2;
  int maxthreads = omp_get_max_threads();
  #pragma omp parallel for num_threads(maxthreads)
  for (i = 1; i < CRITICAL_SIZE; i++) {
    if (a[i] > max) {
      #pragma omp critical
      {
        // compare a[i] and max again because max 
        // could have been changed by another thread after 
        // the comparison outside the critical section
        if (a[i] > max) {
          max = a[i];
        }
        max += busysleep(omp_get_thread_num());
      }
    }
  }
  return max;
}

int critical_named() {
  int i;
  int max;
  int a[CRITICAL_SIZE];

  for (i = 0; i < CRITICAL_SIZE; i++) {
    a[i] = rand();
  }

  max = a[0];
  //int maxthreads = omp_get_num_threads() == 1 ? 1 : 2;
  int maxthreads = omp_get_max_threads();
  #pragma omp parallel for num_threads(maxthreads)
  for (i = 1; i < CRITICAL_SIZE; i++) {
    if (a[i] > max) {
      #pragma omp critical (accumulator)
      {
        // compare a[i] and max again because max 
        // could have been changed by another thread after 
        // the comparison outside the critical section
        if (a[i] > max) {
          max = a[i];
        }
        max += busysleep(omp_get_thread_num());
      }
    }
  }
  return max;
}

void myread(int *data) {
  *data = 1;
}

void process(int *data) {
  (*data)++;
}

int flush() {
  int data;
  int flag = 0;

  int maxthreads = omp_get_num_threads() == 1 ? 1 : 2;
  //int maxthreads = omp_get_num_threads();
  #pragma omp parallel sections default(shared)
  {
    #pragma omp section
    {
      myread(&data);
      #pragma omp flush(data)
      flag = 1;
      #pragma omp flush(flag)
      // Do more work.
    }
    #pragma omp section 
    {
      while (!flag) {
        #pragma omp flush(flag)
      }
      #pragma omp flush(data)
      process(&data);
    }
  }
  return data;
}

int fortest() {
   int i, nRet = 0, nSum = 0, nStart = 0, nEnd = 10;
   int nThreads = 0, nTmp = 10;
   unsigned uTmp = 55;
   int nSumCalc = uTmp;

   if (nTmp < 0)
      nSumCalc = -nSumCalc;

   #pragma omp parallel default(none) private(i) shared(nSum, nThreads, nStart, nEnd)
   {
      #pragma omp master
      nThreads = omp_get_num_threads();

      #pragma omp for
      for (i=nStart; i<=nEnd; ++i) {
            #pragma omp atomic
            nSum += i;
      }
   }
   return nSum;
}

void foo() {
  printf("%d In foo\n", omp_get_thread_num());
}

int parallelfor() {
   int i, nStart = 0, nEnd = 10;
   #pragma omp parallel for
   for (i=nStart; i<=nEnd; ++i) {
     foo();
   }
   return 0;
}

int parallelfor_static() {
   int i, nStart = 0, nEnd = 10;
   // the ordered parameter forces the static scheduler
   #pragma omp parallel for schedule(static) ordered
   for (i=nStart; i<=nEnd; ++i) {
     foo();
   }
   return 0;
}

int parallelfor_dynamic() {
   int i, nStart = 0, nEnd = 10;
   #pragma omp parallel for schedule(dynamic)
   for (i=nStart; i<=nEnd; ++i) {
     foo();
   }
   return 0;
}

int parallelfor_runtime() {
   int i, nStart = 0, nEnd = 10;
   #pragma omp parallel for schedule(runtime)
   for (i=nStart; i<=nEnd; ++i) {
     foo();
   }
   return 0;
}

int master( ) 
{
  int a[5], i;

   #pragma omp parallel
   {
     // Perform some computation.
     #pragma omp for
     for (i = 0; i < 5; i++)
       a[i] = i * i;
        
     // Print intermediate results.
     #pragma omp master
     for (i = 0; i < 5; i++) {
       printf("a[%d] = %d\n", i, a[i]);
	   fflush(stdout);
	 }
        
     // Wait.
     #pragma omp barrier
        
     // Continue with the computation.
     #pragma omp for
     for (i = 0; i < 5; i++)
       a[i] += i;
  }
  return a[3];
}

static float a[1000], b[1000], c[1000];

int test(int first, int last) 
{
  int i;
#pragma omp for ordered
  for (i = first; i <= last; ++i) {
    // Do something here.
    if (i % 2) 
    {
#pragma omp ordered 
      printf("test() iteration %d, thread %d\n", i, omp_get_thread_num());
      fflush(stdout);
    }
  }
  return i;
}

void test2(int iter) 
{
  #pragma omp ordered
  printf("test2() iteration %d, thread %d\n", iter, omp_get_thread_num());
  fflush(stdout);
}

int ordered( ) 
{
  int i;
#pragma omp parallel shared(i)
  {
    i = test(1, 8);
#pragma omp for ordered schedule(dynamic)
    for (i = 0 ; i < 5 ; i++)
      test2(i);
  }
  return i;
}

int sections() {
  #pragma omp parallel sections
  {
    {
      printf("Hello from thread %d\n", omp_get_thread_num());
      fflush(stdout);
    }
    #pragma omp section
    {
      printf("Hello from thread %d\n", omp_get_thread_num());
      fflush(stdout);
    }
  }
  return 1;
}

int single( ) 
{
  int a[5], i;

   #pragma omp parallel
   {
     // Perform some computation.
     #pragma omp for
     for (i = 0; i < 5; i++)
       a[i] = i * i;
        
     // Print intermediate results.
     #pragma omp single
     for (i = 0; i < 5; i++) {
       printf("a[%d] = %d\n", i, a[i]);
       fflush(stdout);
     }
        
     // Wait.
     #pragma omp barrier
        
     // Continue with the computation.
     #pragma omp for
     for (i = 0; i < 5; i++)
       a[i] += i;
  }
  return a[3];
}

#if !defined(TAU_MPC)
int fib(int n) {
  int x,y;
  if (n<2) return n;
  #pragma omp task untied shared(x)
  { x = fib(n-1); }
  #pragma omp task untied shared(y)
  { y = fib(n-2); }
  #pragma omp taskwait
  //printf("%d: fib(%d)=%d\n", omp_get_thread_num(), n, x+y); fflush(stdout);
  return x+y;
}

int fibouter(int n) {
  int answer = 0;
  #pragma omp parallel shared(answer)
  {
    #pragma omp single 
    {
      #pragma omp task shared(answer) 
      {
	    answer = fib(n);
      }
    }
  }
  return answer;
}
#endif

int main (int argc, char *argv[]) 
{
  printf("Main...\n");
  fflush(stdout);
#if 0
#endif
  do_work();
  printf ("\n\nDoing atomic: %d\n\n", atomic()); fflush(stdout);
  printf ("\n\nDoing barrier: %d\n\n", barrier()); fflush(stdout);
  printf ("\n\nDoing fortest: %d\n\n", fortest()); fflush(stdout);
#if !defined(TAU_IBM_OMPT)
  // IBM doesn't handle the flush test well.
  printf ("\n\nDoing flush: %d\n\n", flush()); fflush(stdout);
#endif
  printf ("\n\nDoing master: %d\n\n", master()); fflush(stdout);
#if !defined(TAU_OPEN64ORC) && !defined(TAU_IBM_OMPT) && !defined(TAU_MPC)
  // OpenUH and IBM don't handle the ordered test well.
  printf ("\n\nDoing ordered: %d\n\n", ordered()); fflush(stdout);
#endif
  printf ("\n\nDoing sections: %d\n\n", sections()); fflush(stdout);
  printf ("\n\nDoing single: %d\n\n", single()); fflush(stdout);
  printf ("\n\nDoing critical: %d\n\n", critical()); fflush(stdout);
  printf ("\n\nDoing critical named: %d\n\n", critical_named()); fflush(stdout);
  printf ("\n\nDoing parallelfor: %d\n\n", parallelfor()); fflush(stdout);
  printf ("\n\nDoing parallelfor_dynamic: %d\n\n", parallelfor_dynamic()); fflush(stdout);
#if !defined(TAU_MPC)
  printf ("\n\nDoing parallelfor_static: %d\n\n", parallelfor_static()); fflush(stdout);
  printf ("\n\nDoing parallelfor_runtime: %d\n\n", parallelfor_runtime()); fflush(stdout);
  printf ("\n\nDoing tasks: %d\n\n", fibouter(20)); fflush(stdout);
#endif
#if 0
#endif

  printf ("Done.\n");
  // sleep, so the other threads can finish.
  sleep(1);
  fflush(stdout);

  return 0;
}

