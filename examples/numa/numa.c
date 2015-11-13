// Based on numatest.cpp by James Brock
// http://stackoverflow.com/questions/7259363/measuring-numa-non-uniform-memory-access-no-observable-asymmetry-why
//
// Changes by Andreas Kloeckner, 10/2012:
// - Rewritten in C + OpenMP
// - Rewritten by Sameer Shende, 11/2015 for TAU instrumentation
// - Added contention tests

#include <numa.h>
#include <stdio.h>
#include <assert.h>
#include "timing.h"

#ifdef TAU_MPI
#include <mpi.h>
int rank;
#endif /* TAU_MPI */

////////////////////////////////////////////////////////////////
double measure_access(void *x, size_t array_size, size_t ntrips)
{
  timestamp_type t1;
  get_timestamp(&t1);

  size_t i, j;
  for (i = 0; i<ntrips; ++i)
    for(j = 0; j<array_size; ++j)
    {
      *(((char*)x) + ((j * 1009) % array_size)) += 1;
    }

  timestamp_type t2;
  get_timestamp(&t2);

  return timestamp_diff_in_seconds(t1, t2);
}

////////////////////////////////////////////////////////////////
void access_local( char *x, int node_id) {
  const size_t cache_line_size = 64;
  const size_t array_size = 1000*1000*1000;
  size_t ntrips = 2;
        x = (char *) numa_alloc_onnode(array_size, node_id);
        //x = (char *) malloc(array_size);
        double t = measure_access(x, array_size, ntrips);
#ifdef TAU_MPI
  printf("Rank = %d ", rank);
#endif /* TAU_MPI */
        printf("sequential core %d -> core 0 : BW %g MB/s\n",
            node_id, array_size*ntrips*cache_line_size / t / 1e6);
        numa_free(x, array_size);
}

////////////////////////////////////////////////////////////////
void access_remote( char *x, int node_id) {
  const size_t cache_line_size = 64;
  const size_t array_size = 1000*1000*1000;
  size_t ntrips = 2;
        x = (char *) numa_alloc_onnode(array_size, node_id);
        //x = (char *) malloc(array_size);
        double t = measure_access(x, array_size, ntrips);
#ifdef TAU_MPI
  printf("Rank = %d ", rank);
#endif /* TAU_MPI */
        printf("sequential core %d -> core 0 : BW %g MB/s\n",
            node_id, array_size*ntrips*cache_line_size / t / 1e6);
        numa_free(x, array_size);
}

////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
#ifdef TAU_MPI
  MPI_Init(&argc, &argv);
#endif /* TAU_MPI  */
  int num_cpus = numa_num_task_cpus();
  size_t i;
#ifdef TAU_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("Rank = %d ", rank);
#endif /* TAU_MPI */
  printf("num cpus: %d  ", num_cpus);

  printf("numa available: %d\n", numa_available());

  char *x;

  access_local(x, 0);
  access_remote(x, 1);

#ifdef TAU_MPI
  MPI_Finalize();
#endif /* TAU_MPI */

  return 0;
}
