/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2007  						   	   **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
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

#define SYNC_LOOP_COUNT 10

double& TheTauTraceBeginningOffset() {
  static double offset = 0.0;
  return offset;
}

double& TheTauTraceSyncOffset() {
  static double offset = -1.0;
  return offset;
}

bool& TheTauTraceSyncOffsetSet() {
  static bool value = false;
  return value;
}



double TauSyncAdjustTimeStamp(double timestamp) {
  if (TheTauTraceSyncOffsetSet() == false) {
    // return 0 until sync'd
    return 0.0;
  }

  timestamp = timestamp - TheTauTraceBeginningOffset() + TheTauTraceSyncOffset();
  return timestamp;
}

static double getPreSyncTime(int tid = 0) { 
#ifdef TAU_MULTIPLE_COUNTERS
  // counter 0 is the one we use
  double value = MultipleCounterLayer::getSingleCounter(tid, 0);
#else
  double value = RtsLayer::getUSecD(tid);
#endif
  return value - TheTauTraceBeginningOffset();
}


static double masterServeOffset(int slave) {
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
    PMPI_Send(NULL, 0, MPI_INT, slave, 1, MPI_COMM_WORLD);
    PMPI_Recv(NULL, 0, MPI_INT, slave, 2, MPI_COMM_WORLD, &stat);
    trecv[i] = getPreSyncTime();
  }

  // find minimum ping-pong time
  pingpong_time = trecv[0] - tsend[0];
  min = 0;
  for (i = 1; i < lcount; i++) {
    if ((trecv[i] - tsend[i]) < pingpong_time) {
      pingpong_time = (trecv[i] - tsend[i]);
      min = i;
    }
  }

  sync_time = tsend[min] + (pingpong_time / 2);

  // send index of minimum ping-pong
  PMPI_Send(&min, 1, MPI_INT, slave, 3, MPI_COMM_WORLD);
  // send sync_time
  PMPI_Send(&sync_time, 1, MPI_DOUBLE, slave, 4, MPI_COMM_WORLD);

  // master has no offset
  return 0.0;
}


static double slaveDetermineOffset(int master, int rank) {
  int i, min;
  double tsendrecv[SYNC_LOOP_COUNT];
  double sync_time;
  MPI_Status stat;

  // perform ping-pong loop
  for (i = 0; i < SYNC_LOOP_COUNT; i++) {
    PMPI_Recv(NULL, 0, MPI_INT, master, 1, MPI_COMM_WORLD, &stat);
    tsendrecv[i] = getPreSyncTime();
    PMPI_Send(NULL, 0, MPI_INT, master, 2, MPI_COMM_WORLD);
  }

  // recieve the index of the ping-pong with the lowest time
  PMPI_Recv(&min, 1, MPI_INT, master, 3, MPI_COMM_WORLD,  &stat);
  // recieve the sync_time from the master
  PMPI_Recv(&sync_time, 1, MPI_DOUBLE, master, 4, MPI_COMM_WORLD, &stat);

  double ltime = tsendrecv[min];
  return sync_time - ltime;
}




static double determineOffset(int rank, int size) {
  int i;
  double offset;

  offset = 0.0;

  PMPI_Barrier(MPI_COMM_WORLD);

  for (i = 1; i < size; i++) {
    PMPI_Barrier(MPI_COMM_WORLD);
    if (rank == i ){
      offset = slaveDetermineOffset(0, i);
    } else if (rank == 0) {
      offset = masterServeOffset(i);
    }
  }

  PMPI_Barrier(MPI_COMM_WORLD);


  return offset;
}



// The MPI_Init wrapper calls this routine
extern "C" void TauSyncClocks(int rank, int size) {

  PMPI_Barrier(MPI_COMM_WORLD);
  printf ("TAU: Clock Synchonization active on node : %d\n", rank);
  //   printf ("Zero offset for node %d = %.16G\n", rank, TheTauTraceBeginningOffset());

  // clear counter to zero, since the times might be wildly different (LINUX_TIMERS)
  // we reset to zero so that the offsets won't be so large as to give us negative numbers
  // on some nodes.  This also allows us to easily use 0 before MPI_Init.
  TheTauTraceBeginningOffset() = getPreSyncTime();
  PMPI_Barrier(MPI_COMM_WORLD);


  double syncOffset = determineOffset(rank, size);
  //   printf ("TAU: Sync Offset for node %d = %.16G\n", rank, syncOffset);

  TheTauTraceSyncOffset() = syncOffset;
  TheTauTraceSyncOffsetSet() = true;

}
