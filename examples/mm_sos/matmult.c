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

#include "matmult_initialize.h"

#ifdef TAU_MPI
int provided;
#include <mpi.h>
/* NOTE: MPI is just used to spawn multiple copies of the kernel to different ranks.
This is not a parallel implementation */
#endif /* TAU_MPI */

#ifdef PTHREADS
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
/*** NOTE THE ATTR INITIALIZER HERE! ***/
pthread_mutex_t mutexsum;
#endif /* PTHREADS */

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 512
#endif
#define ITERATIONS 3

#define NRA MATRIX_SIZE                 /* number of rows in matrix A */
#define NCA MATRIX_SIZE                 /* number of columns in matrix A */
#define NCB MATRIX_SIZE                 /* number of columns in matrix B */

double** allocateMatrix(int rows, int cols) {
  int i;
  double **matrix = (double**)malloc((sizeof(double*)) * rows);
  for (i=0; i<rows; i++) {
    matrix[i] = (double*)malloc((sizeof(double)) * cols);
  }
  return matrix;
}

void freeMatrix(double** matrix, int rows, int cols) {
  int i;
  for (i=0; i<rows; i++) {
    free(matrix[i]);
  }
  free(matrix);
}

#ifdef APP_USE_INLINE_MULTIPLY
__inline double multiply(double a, double b) {
	return a * b;
}
#endif /* APP_USE_INLINE_MULTIPLY */

#if 0
// cols_a and rows_b are the same value
void compute_nested(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  int i,j,k;
  double tmp = 0.0;
//num_threads(2)
#pragma omp parallel private(i) shared(a,b,c) 
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
#pragma omp for nowait schedule(dynamic,1)
    for (i=0; i<rows_a; i++) {
//num_threads(2)
#pragma omp parallel private(i,j,k) shared(a,b,c) 
      {
#pragma omp for nowait schedule(dynamic,1)
        for (k=0; k<cols_a; k++) {
          for(j=0; j<cols_b; j++) {
#ifdef APP_USE_INLINE_MULTIPLY
              c[i][j] += multiply(a[i][k], b[k][j]);
#else 
              tmp = a[i][k];
			  tmp = tmp * b[k][j];
              c[i][j] += tmp;
#endif 
            }
          }
      }
    }
  }   /*** End of parallel region ***/
}
#endif

// cols_a and rows_b are the same value
void compute(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
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

void compute_interchange(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
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

double do_work(void) {
  double **a,           /* matrix A to be multiplied */
  **b,           /* matrix B to be multiplied */
  **c;           /* result matrix C */
  a = allocateMatrix(NRA, NCA);
  b = allocateMatrix(NCA, NCB);
  c = allocateMatrix(NRA, NCB);  

/*** Spawn a parallel region explicitly scoping all variables ***/

  initialize(a, NRA, NCA);
  initialize(b, NCA, NCB);
  initialize(c, NRA, NCB);

  compute(a, b, c, NRA, NCA, NCB);
#if defined(TAU_OPENMP)
#if 0
  //if (omp_get_nested()) {
    compute_nested(a, b, c, NRA, NCA, NCB);
  //}
#endif
#endif
  compute_interchange(a, b, c, NRA, NCA, NCB);

  double result = c[0][1];

  freeMatrix(a, NRA, NCA);
  freeMatrix(b, NCA, NCB);
  freeMatrix(c, NCA, NCB);

  return result;
}

#ifdef PTHREADS
int busy_sleep() {
  int i, sum = 0;
  for (i = 0 ; i < 100000000 ; i++) {
    sum = sum+i;
  }
  return sum;
}

void * threaded_func(void *data)
{
  int rc;
  int sum = 0;
  // compute
  int i;
  for (i = 0 ; i < ITERATIONS ; i++) {
  do_work();
  }

#ifdef APP_DO_LOCK_TEST
  // test locking - sampling should catch this
  if ((rc = pthread_mutex_lock(&mutexsum)) != 0)
  {
    errno = rc;
    perror("thread lock error");
    exit(1);
  }
  fprintf(stderr,"Thread 'sleeping'...\n"); fflush(stderr);
  sum += busy_sleep();
  fprintf(stderr,"Thread 'awake'...\n"); fflush(stderr);
  if ((rc = pthread_mutex_unlock(&mutexsum)) != 0)
  {
    errno = rc;
    perror("thread unlock error");
    exit(1);
  }
  pthread_exit((void*) 0);
#endif // APP_DO_LOCK_TEST
  return NULL;
}
#endif // PTHREADS

int main (int argc, char *argv[]) 
{

#ifdef PTHREADS
  int ret;
  pthread_attr_t  attr;
  pthread_t       tid1, tid2, tid3;
  pthread_mutexattr_t Attr;
  pthread_mutexattr_init(&Attr);
#ifndef TAU_CRAYCNL
  pthread_mutexattr_settype(&Attr, PTHREAD_MUTEX_ERRORCHECK);
#endif /* TAU_CRAYCNL */
  if (pthread_mutex_init(&mutexsum, &Attr)) {
   printf("Error while using pthread_mutex_init\n");
  }
#endif /* PTHREADS */

#ifdef TAU_MPI
  int rc = MPI_SUCCESS;
  int rank = 0;
  int comm_size = 0;
#if defined(PTHREADS)
  rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    printf("MPI_Init_thread: provided = %d, MPI_THREAD_MULTIPLE=%d\n", provided, MPI_THREAD_MULTIPLE);
  }
#elif defined(TAU_OPENMP)
  rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    printf("MPI_Init_thread: provided = %d, MPI_THREAD_FUNNELED=%d\n", provided, MPI_THREAD_FUNNELED);
  }
#else
  rc = MPI_Init(&argc, &argv); 
#endif /* THREADS */
  if (rc != MPI_SUCCESS) {
    char *errorstring;
    int length = 0;
    MPI_Error_string(rc, errorstring, &length);
    printf("Error: MPI_Init failed, rc = %d\n%s\n", rc, errorstring);
    exit(1);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
#endif /* TAU_MPI */

#ifdef PTHREADS
  ret = pthread_create(&tid1, NULL, threaded_func, NULL);
  if (ret) {
    printf("Error: pthread_create (1) fails ret = %d\n", ret);
    exit(1);
  }   

  ret = pthread_create(&tid2, NULL, threaded_func, NULL);
  if (ret) {
    printf("Error: pthread_create (2) fails ret = %d\n", ret);
    exit(1);
  }   

  ret = pthread_create(&tid3, NULL, threaded_func, NULL);
  if (ret) {
    printf("Error: pthread_create (3) fails ret = %d\n", ret);
    exit(1);
  }   

#endif /* PTHREADS */

#ifdef TAU_MPI
    // create a communicator
    /* The code above only works with 4 or more processes!! */
    if (comm_size >=4 ) {
      MPI_Group group_world, odd_group, even_group, diff_group, 
                union_group, inter_group, re_group, ri_group;
      int j, Neven, Nodd, members[8], ierr;

      MPI_Comm_group(MPI_COMM_WORLD, &group_world);
      MPI_Comm world_comm;
      MPI_Comm_create(MPI_COMM_WORLD, group_world, &world_comm);

      Neven = (comm_size+1)/2;    /* processes of MPI_COMM_WORLD are divided */
      Nodd = comm_size - Neven;   /* into odd- and even-numbered groups */
      for (j=0; j < Neven; j++) {   /* "members" determines members of even_group */
        members[j] = 2*j;
      };
    
      MPI_Group_incl(group_world, Neven, members, &even_group);
      MPI_Group_excl(group_world, Neven, members, &odd_group);
      MPI_Comm even_comm;
      MPI_Comm odd_comm;
      MPI_Comm_create(MPI_COMM_WORLD, even_group, &even_comm);
      MPI_Comm_create(MPI_COMM_WORLD, odd_group, &odd_comm);
      MPI_Group_difference(group_world, even_group, &diff_group);
      MPI_Group_intersection(group_world, odd_group, &inter_group);
      MPI_Group_union(group_world, odd_group, &union_group);
      int range[2][3] = {{0,1,1},{2,3,1}};
      MPI_Group_range_excl(group_world, 2, range, &re_group);
      MPI_Group_range_incl(group_world, 2, range, &ri_group);
      int ranks[2] = {0,1};
      int ranks_out[2] = {0};
      MPI_Group_translate_ranks(group_world, 2, ranks, union_group, ranks_out);
    } 

#endif /* TAU_MPI */

/* On thread 0: */
  int i;
  for (i = 0 ; i < ITERATIONS ; i++) {
  printf("%d.", i);fflush(stdout);
  do_work();
  }

#ifdef PTHREADS
  ret = pthread_join(tid1, NULL);
  if (ret) {
    printf("Error: pthread_join (1) fails ret = %d\n", ret);
    exit(1);
  }   

  ret = pthread_join(tid2, NULL);
  if (ret) {
    printf("Error: pthread_join (2) fails ret = %d\n", ret);
    exit(1);
  }   

  ret = pthread_join(tid3, NULL);
  if (ret) {
    printf("Error: pthread_join (3) fails ret = %d\n", ret);
    exit(1);
  }   

  pthread_mutex_destroy(&mutexsum);
#endif /* PTHREADS */

#ifdef TAU_MPI
  MPI_Finalize();
#endif /* TAU_MPI */
  printf ("Done.\n");

  return 0;
}

