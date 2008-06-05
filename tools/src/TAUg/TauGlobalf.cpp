/****************************************************************************
**			TAU Portable Profiling Package			                       **
**			http://www.cs.uoregon.edu/research/tau	                       **
*****************************************************************************
**    Copyright 1997-2008                                                  **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/

/***************************************************************************
**  File            : TauGlobalf.cpp                                      **
**  Description     : TAU Global (TAUg) implementation file.              **
**  Author          : Kevin Huck                                          **
**  Contact         : khuck@cs.uoregon.edu                                **
**  Documentation   : See http://tau.uoregon.edu                          **
***************************************************************************/

#include "TauGlobalf.h"
#include <stdio.h>
#include <math.h>
#include <Profile/Profiler.h>
#include "mpi.h"

#include <iostream>

#define TAU_COMMUNICATORS 8

void tau_register_view_(void)
{
   	int viewID = 0;
	TAU_REGISTER_VIEW("rhs", &viewID);
}

void tau_register_communicator_(void)
{
	int commID = 0;
	int numprocs = 0;
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

	for (int mod = 0 ; mod < TAU_COMMUNICATORS ; mod++) {
	  int members[numprocs/TAU_COMMUNICATORS ];
	  int index = mod;
	  for (int i = 0 ; i < numprocs/TAU_COMMUNICATORS ; i++) {
		  members[i] = index;
	      index += TAU_COMMUNICATORS;
	  }
	  TAU_REGISTER_COMMUNICATOR(members, numprocs/TAU_COMMUNICATORS, &commID);
	}

}

void tau_get_global_data_(void)
{
	double *mainTime;
	int timeSize = 0;
   	int viewID = 1;
	int commID = 1;
	int myid = 0;
	int numprocs = 0;
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

	for (int mod = 0 ; mod < TAU_COMMUNICATORS ; mod++) {
	  commID = mod+1;
	  if (myid % TAU_COMMUNICATORS == mod) {
		  TAU_GET_GLOBAL_DATA(viewID, commID, TAU_ALL_TO_ALL, mod, &mainTime, &timeSize);
		  /*
		  if (myid == mod) {
			  printf ("%d timeSize: %d ", myid, timeSize);
			  for (int i = 0 ; i < timeSize ; i++) {
				  printf ("mainTime [%d] = %f ", i, mainTime[i]);
			  }
			  printf("\n");
		  }
		  */
	  }
	}
}

