#include <mpi.h>        /* MPI Library                                            */
	#include <omp.h>        /* OpenMP Library                                         */
	#include <stdio.h>      /* printf()                                               */
	#include <stdlib.h>     /* EXIT_SUCCESS                                           */
#define NRA 100                 /* number of rows in matrix A */
#define NCA 100                 /* number of columns in matrix A */
#define NCB 100                  /* number of columns in matrix B */
int foo(int loop){
int i = 0;
int f = 0;
for(i=0;i<loop; i++){
f = i+1;
}
}
	int main (int argc, char *argv[]) {

	    /* Parameters of MPI.                                                         */
	    int M_N;                             /* number of MPI ranks                   */
	    int M_ID;                            /* MPI rank ID                           */
	    int rtn_val;                         /* return value                          */
	    char name[128];                      /* MPI_MAX_PROCESSOR_NAME == 128         */
	    int namelen;

	    /* Parameters of OpenMP.                                                      */
	    int O_P;                             /* number of OpenMP processors           */
	    int O_T;                             /* number of OpenMP threads              */
	    int O_ID;                            /* OpenMP thread ID                      */

	    /* Initialize MPI.                                                            */
	    /* Construct the default communicator MPI_COMM_WORLD.                         */
	    rtn_val = MPI_Init(&argc,&argv);

	    /* Get a few MPI parameters.                                                  */
	    rtn_val = MPI_Comm_size(MPI_COMM_WORLD,&M_N);    /* get number of MPI ranks   */
	    rtn_val = MPI_Comm_rank(MPI_COMM_WORLD,&M_ID);   /* get MPI rank ID           */
	    MPI_Get_processor_name(name,&namelen);
	    printf("name:%s   M_ID:%d  M_N:%d\n", name,M_ID,M_N);

	    /* Get a few OpenMP parameters.                                               */
	    O_P  = omp_get_num_procs();          /* get number of OpenMP processors       */
	    O_T  = omp_get_num_threads();        /* get number of OpenMP threads          */
	    O_ID = omp_get_thread_num();         /* get OpenMP thread ID                  */
	    printf("name:%s   M_ID:%d  O_ID:%d  O_P:%d  O_T:%d\n", name,M_ID,O_ID,O_P,O_T);


	    /* PARALLEL REGION                                                            */
	    /* Thread IDs range from 0 through omp_get_num_threads()-1.                   */
	    /* We execute identical code in all threads (data parallelization).          
	    #pragma omp parallel private(O_ID)
	    {
	    O_ID = omp_get_thread_num();          /* get OpenMP thread ID               /
	    MPI_Get_processor_name(name,&namelen);
	    printf("parallel region:       name:%s M_ID=%d O_ID=%d\n", name,M_ID,O_ID);
	    }*/

int     tid, nthreads, i, j, k;
double  a[NRA][NCA],           /* matrix A to be multiplied */
        b[NCA][NCB],           /* matrix B to be multiplied */
        c[NRA][NCB];           /* result matrix C */


/*** Spawn a parallel region explicitly scoping all variables ***/
#pragma omp parallel shared(a,b,c,nthreads) private(tid,i,j,k)
  {
  tid = omp_get_thread_num();
  if (tid == 0)
    {
    nthreads = omp_get_num_threads();
    printf("Starting matrix multiple example with %d threads\n",nthreads);
    printf("Initializing matrices...\n");
  for (i=0; i<1000000; i++)
          foo(10);
    }
  /*** Initialize matrices ***/
  #pragma omp for schedule (static ) 
  for (i=0; i<NRA; i++)
    for (j=0; j<NCA; j++)
      a[i][j]= i+j;
  #pragma omp for schedule (static )
  for (i=0; i<NCA; i++)
    for (j=0; j<NCB; j++)
      b[i][j]= i*j;
  #pragma omp for schedule (static)
  for (i=0; i<NRA; i++)
    for (j=0; j<NCB; j++)
      c[i][j]= 0;

  /*** Do matrix multiply sharing iterations on outer loop ***/
  /*** Display who does which iterations for demonstration purposes ***/
  printf("Thread %d starting matrix multiply...\n",tid);
  #pragma omp for schedule (static )
  for (i=0; i<NRA; i++)
    {
    for(j=0; j<NCB; j++)
      for (k=0; k<NCA; k++)
        c[i][j] += a[i][k] * b[k][j];
    }
  }   /*** End of parallel region ***/



	    /* Terminate MPI.                                                             */
	    rtn_val = MPI_Finalize();

	    /* Exit master thread.                                                        */
	    printf("name:%s M_ID:%d O_ID:%d   Exits\n", name,M_ID,O_ID);
	    return EXIT_SUCCESS;
	}

