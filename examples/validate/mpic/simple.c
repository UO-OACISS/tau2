/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 1997  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/* This application calculates the value of pi and e using a parallel  */
/* algorithm for integrating a function using Riemann sum. Uses MPI. */
#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <Profile/Profiler.h>

#ifndef M_E
#define M_E         2.7182818284590452354       /* e */
#endif
#ifndef M_PI
#define M_PI        3.14159265358979323846      /* pi */
#endif

double f(double a)
{
    TAU_PROFILE("f()", "double (double)", TAU_USER);
    return (4.0 / (1.0 + a*a));
}

int main(int argc, char* argv[])
{
    int i, n, myid, numprocs, namelen;
    double mySum, h, sum, x;
    double startwtime, timePi, timeE, time1;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    TAU_PROFILE_TIMER(t1, "main-init()", "int (int, char **)",  TAU_USER);
    TAU_PROFILE_INIT(argc,argv);
    MPI_Init(&argc,&argv);
    TAU_PROFILE_START(t1);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(processor_name,&namelen);
    TAU_PROFILE_SET_NODE(myid);

    fprintf(stderr,"Process %d on %s\n", myid, processor_name);

    if (argc > 1)
    {
        sscanf(argv[1], "%d", &n);
    }
    else
    {
        n = 1000000;
    }
    TAU_PROFILE_STOP(t1);

    /*
    // Calculate pi by integrating 4/(1 + x^2) from 0 to 1.
    */	 

    startwtime = MPI_Wtime();

    h   = 1.0 / (double) n;
    sum = 0.0;
    for (i = myid + 1; i <= n; i += numprocs)
    {
        x = h * ((double)i - 0.5);
        sum += f(x);
    }
    mySum = h * sum;

/* REGION A */
    
    MPI_Reduce(&mySum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
	timePi = MPI_Wtime() - startwtime;
	printf("\nFor %d integration intervals:\n", n);
        printf("  pi is approximately %.16f, Error is %.16f\n",
	       sum, fabs(sum - M_PI));
    }

    /*
    // Calculate e by integrating exp(x) from 0 to 1.
    */	 

    startwtime = MPI_Wtime();

    h   = 1.0 / (double) n;
    sum = 0.0;
    for (i = myid + 1; i <= n; i += numprocs)
    {
        x = h * ((double)i - 0.5);
        sum += exp(x);
    }
    mySum = h * sum;

    MPI_Reduce(&mySum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
        sum += 1;		/* integral = e - 1, so add 1 */
	timeE = MPI_Wtime() - startwtime;
        printf("  e is approximately %.16f, Error is %.16f\n",
	       sum, fabs(sum - M_E));
    }

    /*
    // Calculate 1.0 by integrating cos(th) from 0 to pi/2.
    */	 

    startwtime = MPI_Wtime();

    h   = (M_PI / 2.0) / (double) n;
    sum = 0.0;
    for (i = myid + 1; i <= n; i += numprocs)
    {
        x = h * ((double)i - 0.5);
	sum += cos(x);
    }
    mySum = h * sum;

    MPI_Reduce(&mySum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myid == 0)
    {
	time1 = MPI_Wtime() - startwtime;
        printf("  sum is %.16f, Error is %.16f\n\n", sum, fabs(1.0 - sum));
	printf("wall clock time for pi = %f\n", timePi);
	printf("wall clock time for e  = %f\n", timeE);
	printf("wall clock time for 1  = %f\n", time1);
	printf("Total time = %f\n\n", timePi + timeE + time1);
    }

    /*
    // finished
    */

    MPI_Finalize();
    return 0;
}
