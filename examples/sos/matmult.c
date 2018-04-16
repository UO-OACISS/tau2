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

// #define PROFILING_ON
#include "TAU.h"
#include "matmult_initialize.h"

#include <mpi.h>
int provided = MPI_THREAD_SINGLE;
/* NOTE: MPI is just used to spawn multiple copies of the kernel to different ranks.
This is not a parallel implementation */

#ifdef PTHREADS
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
/*** NOTE THE ATTR INITIALIZER HERE! ***/
pthread_mutex_t mutexsum;
static pthread_barrier_t barrier;
#ifndef PTHREAD_MUTEX_ERRORCHECK
#define PTHREAD_MUTEX_ERRORCHECK 0
#endif
#endif /* PTHREADS */

#ifndef MATRIX_SIZE
#define MATRIX_SIZE 256
#endif

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

// cols_a and rows_b are the same value
void compute(double **a, double **b, double **c, int rows_a, int cols_a, int cols_b) {
  int i,j,k;
#pragma omp parallel private(i,j,k) shared(a,b,c)
  {
    /*** Do matrix multiply sharing iterations on outer loop ***/
    /*** Display who does which iterations for demonstration purposes ***/
#pragma omp for nowait
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
#pragma omp for nowait
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
  compute_interchange(a, b, c, NRA, NCA, NCB);

  double result = c[0][1];

  freeMatrix(a, NRA, NCA);
  freeMatrix(b, NCA, NCB);
  freeMatrix(c, NCA, NCB);

  return result;
}

void * threaded_func(void *data)
{
    int * maxi = (int *)(data);
  for (int i = 0 ; i < *maxi ; i++) {
#ifdef PTHREADS
    int s = pthread_barrier_wait(&barrier);
#endif
    do_work();
  }
#ifdef PTHREADS
  pthread_exit(NULL);
#endif
  return NULL;
}

int main (int argc, char *argv[]) 
{
  int rc = MPI_SUCCESS;
#if defined(PTHREADS)
  rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  printf("MPI_Init_thread: provided = %d, MPI_THREAD_MULTIPLE=%d\n", provided, MPI_THREAD_MULTIPLE);
#elif defined(TAU_OPENMP)
  rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  printf("MPI_Init_thread: provided = %d, MPI_THREAD_FUNNELED=%d\n", provided, MPI_THREAD_FUNNELED);
#else
  rc = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
#endif /* THREADS */
  if (provided == MPI_THREAD_MULTIPLE) { 
    printf("provided is MPI_THREAD_MULTIPLE\n");
  } else if (provided == MPI_THREAD_FUNNELED) { 
    printf("provided is MPI_THREAD_FUNNELED\n");
  }
  if (rc != MPI_SUCCESS) {
    char *errorstring;
    int length = 0;
    MPI_Error_string(rc, errorstring, &length);
    printf("Error: MPI_Init failed, rc = %d\n%s\n", rc, errorstring);
    exit(1);
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int comm_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    // get the number of cores
    unsigned int ncores = sysconf(_SC_NPROCESSORS_ONLN);
    unsigned int nthreads = (ncores / comm_size) -1;
    // for debugging, use just one thread
    nthreads = 1;
    if (rank == 0) { printf("Running with %d processes, %d cores, %d threads per core\n", comm_size, ncores, nthreads+1); }

    int number, count, tag;
    count = 1;
    tag = 1984;
    if (comm_size > 1 && comm_size % 2 == 0) {
    	if (rank % 2 == 0) {
        	number = -1;
        	MPI_Send(&number, count, MPI_INT, rank+1, tag, MPI_COMM_WORLD);
        	printf("Process %d sent number     %d to process   %d\n", rank, number, rank+1);
    	} else {
        	MPI_Recv(&number, count, MPI_INT, rank-1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        	printf("Process %d received number %d from process %d\n", rank, number, rank-1);
    	}
    }

    // create a communicator
    MPI_Group group_world, odd_group, even_group;;
    int i, Neven, Nodd, members[8], ierr;

    MPI_Comm_group(MPI_COMM_WORLD, &group_world);

    Neven = (comm_size+1)/2;    /* processes of MPI_COMM_WORLD are divided */
    Nodd = comm_size - Neven;   /* into odd- and even-numbered groups */
    for (i=0; i < Neven; i++) {   /* "members" determines members of even_group */
      members[i] = 2*i;
    };

    MPI_Group_incl(group_world, Neven, members, &even_group);
    MPI_Group_excl(group_world, Neven, members, &odd_group);
    MPI_Comm even_comm;
    MPI_Comm odd_comm;
    MPI_Comm_create(MPI_COMM_WORLD, even_group, new_comm);
    MPI_Comm_create(MPI_COMM_WORLD, odd_group, new_comm);

#ifdef PTHREADS
  int ret;
  pthread_attr_t  attr;
  //pthread_t       tid1, tid2, tid3;
  pthread_t * tid = (pthread_t*)(malloc(sizeof(pthread_t) * nthreads));
  pthread_mutexattr_t Attr;
  pthread_mutexattr_init(&Attr);
  int s = pthread_barrier_init(&barrier, NULL, nthreads+1);
  pthread_mutexattr_settype(&Attr, PTHREAD_MUTEX_ERRORCHECK);
  if (pthread_mutex_init(&mutexsum, &Attr)) {
   printf("Error while using pthread_mutex_init\n");
  }
#endif /* PTHREADS */

  int maxi = 10;

#ifdef PTHREADS
  for (int i = 0 ; i < nthreads ; i++) {
    if (ret = pthread_create(&(tid[i]), NULL, threaded_func, &maxi) ) {
      printf("Error: pthread_create (1) fails ret = %d\n", ret);
      exit(1);
    }   
  }   
#endif /* PTHREADS */

  char * periodic_str = getenv("TAU_SOS_PERIODIC");
  int do_dump = 1;
  if (periodic_str != NULL) {
    do_dump = 0;
  }

/* On thread 0: */
  int i;
  TAU_REGISTER_CONTEXT_EVENT(event, "Iteration count");
  for (i = 0 ; i < maxi ; i++) {
    // for SOS testing purposes...
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) { printf("Iteration %d of %d working...", i, maxi); fflush(stdout); }
    TAU_CONTEXT_EVENT(event, i);
    int s = pthread_barrier_wait(&barrier);
    do_work();
    if (do_dump) {
        if (rank == 0) { printf("Iteration %d of %d Sending data over SOS....", i, maxi); fflush(stdout); }
        Tau_dump();
    }
    if (rank == 0) { printf("Iteration %d of %d done.\n", i, maxi); fflush(stdout); }
  }

    MPI_Barrier(MPI_COMM_WORLD);
    count = 1;
    tag = 1984;
    if (comm_size > 1 && comm_size % 2 == 0) {
    	if (rank % 2 == 0) {
        	number = -1;
        	MPI_Send(&number, count, MPI_INT, rank+1, tag, MPI_COMM_WORLD);
        	printf("Process %d sent number     %d to process   %d\n", rank, number, rank+1);
    	} else {
        	MPI_Recv(&number, count, MPI_INT, rank-1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        	printf("Process %d received number %d from process %d\n", rank, number, rank-1);
    	}
    }


#ifdef PTHREADS 
  for (int i = 0 ; i < nthreads ; i++) {
    if (ret = pthread_join(tid[i], NULL) )
    {
        printf("Error: pthread_join (1) fails ret = %d\n", ret);
        exit(1);
    }   
  }   
  pthread_mutex_destroy(&mutexsum);
#endif /* PTHREADS */

  MPI_Finalize();
  printf ("Done.\n");

  return 0;
}

