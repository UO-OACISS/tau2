/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2007  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
**    Forschungszentrum Juelich                                            **
****************************************************************************/
/****************************************************************************
**	File 		: ClockSync.cpp  				   **
**	Description 	: TAU Profiling Package				   **
**	Author		: Alan Morris					   **
**	Contact		: tau-bugs@cs.uoregon.edu               	   **
**	Documentation	: See http://www.cs.uoregon.edu/research/tau       **
**                                                                         **
**      Description     : Synchronize Clocks using MPI for tracing and     **
**                                                                         **
**      This code is adapted from Kojak, See:                              **
**                                                                         **
**        http://www.fz-juelich.de/zam/kojak/                              **
**        http://icl.cs.utk.edu/kojak/                                     **
**                                                                         **
**                                                                         **
****************************************************************************/


#include <TAU.h>

#include <mpi.h>
#include <stdio.h>
#include <Profile/tau_types.h>


#ifdef TRACING_ON
#ifdef TAU_EPILOG
#include "elg_trc.h"
#else /* TAU_EPILOG */
#define PCXX_EVENT_SRC
#include "Profile/pcxx_events.h"
#endif /* TAU_EPILOG */
#endif /* TRACING_ON */

#define SYNC_LOOP_COUNT 10

double* TAUDECL TheTauTraceBeginningOffset();
int* TAUDECL TheTauTraceSyncOffsetSet();
double* TAUDECL TheTauTraceSyncOffset();
double TAUDECL TAUClockTime(int tid);

long TauUserEvent_GetEventId(void *evt);

/* We're probably going to have to change this for some platforms */
#ifdef TAU_WINDOWS
static long gethostid() {
  int id;
  char hostname[256];/* 255 is max legal DNS name */
  size_t hostname_size = 256;
  struct hostent *hostp;
  struct in_addr addru;/* union for conversion */
  
  (void) gethostname(hostname, hostname_size);
  hostname[hostname_size] = '\0'; /* make sure it is null-terminated */
  
  hostp = gethostbyname(hostname);
  if(hostp == NULL) {
    /* our own name was not found!  punt. */
    id = 0;
  } else {
    /* return first address of host */
    memcpy(&(addru.s_addr), hostp->h_addr_list[0], 4);
    id = addru.s_addr;
  }
  
  return id;
}
#else
#include <unistd.h>
#endif /* TAU_WINDOWS */

static long getUniqueMachineIdentifier() {
  return gethostid();
}

static double getPreSyncTime() { 
  int tid = 0;
  double value = TAUClockTime(tid);
  return value - *TheTauTraceBeginningOffset();
}


static double masterServeOffset(int slave, MPI_Comm comm) {
  int lcount;
  int i, min;
  double tsend[SYNC_LOOP_COUNT], trecv[SYNC_LOOP_COUNT];
  double pingpong_time, sync_time;
  MPI_Status stat;
  lcount = SYNC_LOOP_COUNT;


  if (lcount > SYNC_LOOP_COUNT) lcount = SYNC_LOOP_COUNT;
  /* exchange lcount ping pong messages with slave */

  for (i = 0; i < lcount; i++) {
    tsend[i] = getPreSyncTime();
    PMPI_Send(NULL, 0, MPI_INT, slave, 1, comm);
    PMPI_Recv(NULL, 0, MPI_INT, slave, 2, comm, &stat);
    trecv[i] = getPreSyncTime();
  }

  /* find minimum ping-pong time */
  pingpong_time = trecv[0] - tsend[0];
  min = 0;
  for (i = 1; i < lcount; i++) {
    if ((trecv[i] - tsend[i]) < pingpong_time) {
      pingpong_time = (trecv[i] - tsend[i]);
      min = i;
    }
  }

  sync_time = tsend[min] + (pingpong_time / 2);

  /* send index of minimum ping-pong */
  PMPI_Send(&min, 1, MPI_INT, slave, 3, comm);
  /* send sync_time */
  PMPI_Send(&sync_time, 1, MPI_DOUBLE, slave, 4, comm);

  /* master has no offset */
  return 0.0;
}


static double slaveDetermineOffset(int master, int rank, MPI_Comm comm) {
  int i, min;
  double tsendrecv[SYNC_LOOP_COUNT];
  double sync_time;
  MPI_Status stat;
  double ltime;

  /* perform ping-pong loop */
  for (i = 0; i < SYNC_LOOP_COUNT; i++) {
    PMPI_Recv(NULL, 0, MPI_INT, master, 1, comm, &stat);
    tsendrecv[i] = getPreSyncTime();
    PMPI_Send(NULL, 0, MPI_INT, master, 2, comm);
  }

  /* receive the index of the ping-pong with the lowest time */
  PMPI_Recv(&min, 1, MPI_INT, master, 3, comm,  &stat);
  /* receive the sync_time from the master */
  PMPI_Recv(&sync_time, 1, MPI_DOUBLE, master, 4, comm, &stat);

  ltime = tsendrecv[min];
  return sync_time - ltime;
}



static double getTimeOffset(int rank, int size) {
  int i;
  MPI_Comm machineComm;
  int machineRank;
  int numProcsThisMachine;
  /* inter-machine communicator */
  MPI_Comm interMachineComm;
  int numMachines;
  /* sync rank is the rank within the inter-machine communicator */
  int syncRank;
  double startOffset;
  double offset;

  PMPI_Comm_split(MPI_COMM_WORLD, getUniqueMachineIdentifier() & 0x7FFFFFFF, 0, &machineComm);
  PMPI_Comm_rank(machineComm, &machineRank);
  PMPI_Comm_size(machineComm, &numProcsThisMachine);

  /* create a communicator with one process from each machine */
  PMPI_Comm_split(MPI_COMM_WORLD, machineRank, 0, &interMachineComm);
  PMPI_Comm_rank(interMachineComm, &syncRank);
  PMPI_Comm_size(interMachineComm, &numMachines);

  /* broadcast the associated starting offset */
  startOffset = *TheTauTraceBeginningOffset();
  PMPI_Bcast(&startOffset, 1, MPI_DOUBLE, 0, machineComm);
  *TheTauTraceBeginningOffset() = startOffset;

  offset = 0.0;
  PMPI_Barrier(MPI_COMM_WORLD);

  if (machineRank == 0) {
    for (i = 1; i < numMachines; i++) {
      PMPI_Barrier(interMachineComm);
      if (syncRank == i ){
	offset = slaveDetermineOffset(0, i, interMachineComm);
      } else if (syncRank == 0) {
	offset = masterServeOffset(i, interMachineComm);
      }
    }
  }

  /* broadcast the result to other processes on this machine */
  PMPI_Bcast(&offset, 1, MPI_DOUBLE, 0, machineComm);


  return offset;
}

/* The MPI_Finalize wrapper calls this routine */
void TauSyncFinalClocks(int rank, int size) {
  /* only do this when tracing */
#ifdef TRACING_ON
#ifndef TAU_EPILOG
  double offset = getTimeOffset(rank, size);
  double diff = *TheTauTraceSyncOffset() - offset;
  TAU_REGISTER_EVENT(endOffset, "TauTraceClockOffsetEnd");
  offset = getTimeOffset(rank, size);
  TraceEvent(
	     TauUserEvent_GetEventId(endOffset), 
	     (x_int64) offset, 
	     0, 0, 0);
#endif 
#endif
}

/* The MPI_Init wrapper calls this routine */
void TauSyncClocks(int rank, int size) {
  double offset = 0;

  PMPI_Barrier(MPI_COMM_WORLD);
  printf ("TAU: Clock Synchonization active on node : %d\n", rank);
  /* clear counter to zero, since the times might be wildly different (LINUX_TIMERS)
     we reset to zero so that the offsets won't be so large as to give us negative numbers
     on some nodes.  This also allows us to easily use 0 before MPI_Init. */
  *TheTauTraceBeginningOffset() = getPreSyncTime();

  /* only do this when tracing */
#ifdef TRACING_ON
  TAU_REGISTER_EVENT(beginOffset, "TauTraceClockOffsetStart");
  offset = getTimeOffset(rank, size);
#endif

  *TheTauTraceSyncOffset() = offset;
  *TheTauTraceSyncOffsetSet() = 1;

#ifdef TRACING_ON
#ifndef TAU_EPILOG
  TraceEvent(TauUserEvent_GetEventId(beginOffset), (x_int64) offset, 0, 0, 0);
#endif
#endif

  PMPI_Barrier(MPI_COMM_WORLD);
}
