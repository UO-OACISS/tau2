/*********************************************************************/
/*                      Copyright (C) 2010                           */
/*   University of Oregon          Barcelona Supercomputing Center   */
/*********************************************************************/

/*
 * paraver_mpi_mapping.h : header for tau_convert, for supporting Paraver
 *
 * (c) 2010 Kevin A. Huck
 *
 * Version 3.0
 * Revision Sep 2010 Author Kevin Huck, (kevin.huck@bsc.es)
 */

#include <string.h>

// Paraver event types
#define PARAVER_USER_FUNCTION_TYPE 60000019
#define MPITYPE_PTOP               50000001
#define MPITYPE_COLLECTIVE         50000002
#define MPITYPE_OTHER              50000003
#define MPITYPE_RMA                50000004
#define MPITYPE_COMM               MPITYPE_OTHER
#define MPITYPE_GROUP              MPITYPE_OTHER
#define MPITYPE_TOPOLOGIES         MPITYPE_OTHER
#define MPITYPE_TYPE               MPITYPE_OTHER
#define MPITYPE_IO                 50000005

#define STATE_IDLE                  0  // Idle
#define STATE_RUNNING               1  // Running
#define STATE_NOT_CREATED           2  // Not created
#define STATE_WAITING_A_MESSAGE     3  // Waiting a message
#define STATE_BLOCKING_SEND         4  // Blocking Send
#define STATE_SYNCHRONIZATION       5  // Synchronization
#define STATE_TEST_PROBE            6  // Test/Probe
#define STATE_SCHEDULING_FORK_JOIN  7  // Scheduling and Fork/Join
#define STATE_WAIT_WAITALL          8  // Wait/WaitAll
#define STATE_BLOCKED               9  // Blocked
#define STATE_IMMEDIATE_SEND       10  // Immediate Send
#define STATE_IMMEDIATE_RECV       11  // Immediate Receive
#define STATE_IO                   12  // I/O
#define STATE_GROUP_COMMUNICATION  13  // Group Communication
#define STATE_TRACING_DISABLED     14  // Tracing Disabled
#define STATE_OTHERS               15  // Others
#define STATE_SEND_RECV            16  // Send Receive

typedef struct paraver_map_entry {
	int type;
	int value;
	char *name;
	int state;
} PARAVER_MAP_ENTRY;

#define NUM_MPI_FUNCTIONS 140

static PARAVER_MAP_ENTRY MPI_FUNCTION_MAP[] = {
// point to points
{MPITYPE_PTOP, 33,  "MPI_Bsend", STATE_BLOCKING_SEND},
{MPITYPE_PTOP, 112, "MPI_Bsend_init", STATE_BLOCKING_SEND},
{MPITYPE_PTOP, 40,  "MPI_Cancel", STATE_IDLE},
{MPITYPE_PTOP, 116, "MPI_Recv_init", STATE_WAITING_A_MESSAGE},
{MPITYPE_PTOP, 117, "MPI_Send_init", STATE_IMMEDIATE_SEND},
{MPITYPE_PTOP, 36,  "MPI_Ibsend", STATE_IMMEDIATE_SEND},
{MPITYPE_PTOP, 62,  "MPI_Iprobe", STATE_IDLE},
{MPITYPE_PTOP, 4,   "MPI_Irecv", STATE_IMMEDIATE_RECV},
{MPITYPE_PTOP, 38,  "MPI_Irsend", STATE_IMMEDIATE_SEND},
{MPITYPE_PTOP, 3,   "MPI_Isend", STATE_IMMEDIATE_SEND},
{MPITYPE_PTOP, 37,  "MPI_Issend", STATE_IMMEDIATE_SEND},
{MPITYPE_PTOP, 61,  "MPI_Probe", STATE_IDLE},
{MPITYPE_PTOP, 2,   "MPI_Recv", STATE_WAITING_A_MESSAGE},
{MPITYPE_PTOP, 35,  "MPI_Rsend", STATE_BLOCKING_SEND},
{MPITYPE_PTOP, 121, "MPI_Rsend_init", STATE_BLOCKING_SEND},
{MPITYPE_PTOP, 1,   "MPI_Send", STATE_BLOCKING_SEND},
{MPITYPE_PTOP, 41,  "MPI_Sendrecv", STATE_SEND_RECV},
{MPITYPE_PTOP, 42,  "MPI_Sendrecv_replace", STATE_SEND_RECV},
{MPITYPE_PTOP, 34,  "MPI_Ssend", STATE_BLOCKING_SEND},
{MPITYPE_PTOP, 122, "MPI_Ssend_init", STATE_BLOCKING_SEND},
{MPITYPE_PTOP, 5,   "MPI_Wait", STATE_WAIT_WAITALL},
{MPITYPE_PTOP, 6,   "MPI_Waitall", STATE_WAIT_WAITALL},
{MPITYPE_PTOP, 59,  "MPI_Waitany", STATE_WAIT_WAITALL},
{MPITYPE_PTOP, 60,  "MPI_Waitsome", STATE_WAIT_WAITALL},
{MPITYPE_PTOP, 0,   "End", STATE_RUNNING},
// collectives
{MPITYPE_COLLECTIVE, 17, "MPI_Allgather", STATE_GROUP_COMMUNICATION},
{MPITYPE_COLLECTIVE, 18, "MPI_Allgatherv", STATE_GROUP_COMMUNICATION},
{MPITYPE_COLLECTIVE, 10, "MPI_Allreduce", STATE_GROUP_COMMUNICATION},
{MPITYPE_COLLECTIVE, 11, "MPI_Alltoall", STATE_GROUP_COMMUNICATION},
{MPITYPE_COLLECTIVE, 12, "MPI_Alltoallv", STATE_GROUP_COMMUNICATION},
{MPITYPE_COLLECTIVE, 8,  "MPI_Barrier", STATE_SYNCHRONIZATION},
{MPITYPE_COLLECTIVE, 7,  "MPI_Bcast", STATE_SYNCHRONIZATION},
{MPITYPE_COLLECTIVE, 13, "MPI_Gather", STATE_GROUP_COMMUNICATION},
{MPITYPE_COLLECTIVE, 14, "MPI_Gatherv", STATE_GROUP_COMMUNICATION},
{MPITYPE_COLLECTIVE, 80, "MPI_Reduce_scatter", STATE_GROUP_COMMUNICATION},
{MPITYPE_COLLECTIVE, 9,  "MPI_Reduce", STATE_GROUP_COMMUNICATION},
{MPITYPE_COLLECTIVE, 30, "MPI_Scan", STATE_GROUP_COMMUNICATION},
{MPITYPE_COLLECTIVE, 15, "MPI_Scatter", STATE_GROUP_COMMUNICATION},
{MPITYPE_COLLECTIVE, 16, "MPI_Scatterv", STATE_GROUP_COMMUNICATION},
{MPITYPE_COLLECTIVE, 0,  "End", STATE_RUNNING},
// other
{MPITYPE_OTHER, 78, "MPI_Op_create", STATE_IDLE},
{MPITYPE_OTHER, 79, "MPI_Op_free", STATE_IDLE},
{MPITYPE_OTHER, 81, "MPI_Attr_delete", STATE_IDLE},
{MPITYPE_OTHER, 82, "MPI_Attr_get", STATE_IDLE},
{MPITYPE_OTHER, 83, "MPI_Attr_put", STATE_IDLE},
{MPITYPE_OTHER, 21, "MPI_Comm_create", STATE_IDLE},
{MPITYPE_OTHER, 22, "MPI_Comm_dup", STATE_IDLE},
{MPITYPE_OTHER, 25, "MPI_Comm_free", STATE_IDLE},
{MPITYPE_OTHER, 24, "MPI_Comm_group", STATE_IDLE},
{MPITYPE_OTHER, 19, "MPI_Comm_rank", STATE_IDLE},
{MPITYPE_OTHER, 26, "MPI_Comm_remote_group", STATE_IDLE},
{MPITYPE_OTHER, 27, "MPI_Comm_remote_size", STATE_IDLE},
{MPITYPE_OTHER, 20, "MPI_Comm_size", STATE_IDLE},
{MPITYPE_OTHER, 23, "MPI_Comm_split", STATE_IDLE},
{MPITYPE_OTHER, 28, "MPI_Comm_test_inter", STATE_IDLE},
{MPITYPE_OTHER, 29, "MPI_Comm_compare", STATE_IDLE},
{MPITYPE_OTHER, 84, "MPI_Group_difference", STATE_IDLE},
{MPITYPE_OTHER, 85, "MPI_Group_excl", STATE_IDLE},
{MPITYPE_OTHER, 86, "MPI_Group_free", STATE_IDLE},
{MPITYPE_OTHER, 87, "MPI_Group_incl", STATE_IDLE},
{MPITYPE_OTHER, 88, "MPI_Group_intersection", STATE_IDLE},
{MPITYPE_OTHER, 89, "MPI_Group_rank", STATE_IDLE},
{MPITYPE_OTHER, 90, "MPI_Group_range_excl", STATE_IDLE},
{MPITYPE_OTHER, 91, "MPI_Group_range_incl", STATE_IDLE},
{MPITYPE_OTHER, 92, "MPI_Group_size", STATE_IDLE},
{MPITYPE_OTHER, 93, "MPI_Group_translate_ranks", STATE_IDLE},
{MPITYPE_OTHER, 94, "MPI_Group_union", STATE_IDLE},
{MPITYPE_OTHER, 95, "MPI_Group_compare", STATE_IDLE},
{MPITYPE_OTHER, 96, "MPI_Intercomm_create", STATE_IDLE},
{MPITYPE_OTHER, 97, "MPI_Intercomm_merge", STATE_IDLE},
{MPITYPE_OTHER, 98, "MPI_Keyval_free", STATE_IDLE},
{MPITYPE_OTHER, 99, "MPI_Keyval_create", STATE_IDLE},
{MPITYPE_OTHER, 100, "MPI_Abort", STATE_IDLE},
{MPITYPE_OTHER, 101, "MPI_Error_class", STATE_IDLE},
{MPITYPE_OTHER, 102, "MPI_Errhandler_create", STATE_IDLE},
{MPITYPE_OTHER, 103, "MPI_Errhandler_free", STATE_IDLE},
{MPITYPE_OTHER, 104, "MPI_Errhandler_get", STATE_IDLE},
{MPITYPE_OTHER, 105, "MPI_Error_string", STATE_IDLE},
{MPITYPE_OTHER, 106, "MPI_Errhandler_set", STATE_IDLE},
{MPITYPE_OTHER, 32, "MPI_Finalize", STATE_IDLE},
{MPITYPE_OTHER, 107, "MPI_Get_processor_name", STATE_IDLE},
{MPITYPE_OTHER, 31, "MPI_Init", STATE_IDLE},
{MPITYPE_OTHER, 108, "MPI_Initialized", STATE_IDLE},
{MPITYPE_OTHER, 109, "MPI_Wtick", STATE_IDLE},
{MPITYPE_OTHER, 110, "MPI_Wtime", STATE_IDLE},
{MPITYPE_OTHER, 111, "MPI_Address", STATE_IDLE},
{MPITYPE_OTHER, 113, "MPI_Buffer_attach", STATE_IDLE},
{MPITYPE_OTHER, 114, "MPI_Buffer_detach", STATE_IDLE},
{MPITYPE_OTHER, 115, "MPI_Request_free", STATE_IDLE},
{MPITYPE_OTHER, 118, "MPI_Get_count", STATE_IDLE},
{MPITYPE_OTHER, 119, "MPI_Get_elements", STATE_IDLE},
{MPITYPE_OTHER, 76, "MPI_Pack", STATE_IDLE},
{MPITYPE_OTHER, 120, "MPI_Pack_size", STATE_IDLE},
{MPITYPE_OTHER, 123, "MPI_Start", STATE_IDLE},
{MPITYPE_OTHER, 124, "MPI_Startall", STATE_IDLE},
{MPITYPE_OTHER, 39, "MPI_Test", STATE_TEST_PROBE},
{MPITYPE_OTHER, 125, "MPI_Testall", STATE_TEST_PROBE},
{MPITYPE_OTHER, 126, "MPI_Testany", STATE_TEST_PROBE},
{MPITYPE_OTHER, 127, "MPI_Test_cancelled", STATE_TEST_PROBE},
{MPITYPE_OTHER, 128, "MPI_Testsome", STATE_TEST_PROBE},
{MPITYPE_OTHER, 129, "MPI_Type_commit", STATE_IDLE},
{MPITYPE_OTHER, 130, "MPI_Type_contiguous", STATE_IDLE},
{MPITYPE_OTHER, 131, "MPI_Type_extent", STATE_IDLE},
{MPITYPE_OTHER, 132, "MPI_Type_free", STATE_IDLE},
{MPITYPE_OTHER, 133, "MPI_Type_hindexed", STATE_IDLE},
{MPITYPE_OTHER, 134, "MPI_Type_hvector", STATE_IDLE},
{MPITYPE_OTHER, 135, "MPI_Type_indexed", STATE_IDLE},
{MPITYPE_OTHER, 136, "MPI_Type_lb", STATE_IDLE},
{MPITYPE_OTHER, 137, "MPI_Type_size", STATE_IDLE},
{MPITYPE_OTHER, 138, "MPI_Type_struct", STATE_IDLE},
{MPITYPE_OTHER, 139, "MPI_Type_ub", STATE_IDLE},
{MPITYPE_OTHER, 140, "MPI_Type_vector", STATE_IDLE},
{MPITYPE_OTHER, 77, "MPI_Unpack", STATE_IDLE},
{MPITYPE_OTHER, 45, "MPI_Cart_coords", STATE_IDLE},
{MPITYPE_OTHER, 43, "MPI_Cart_create", STATE_IDLE},
{MPITYPE_OTHER, 46, "MPI_Cart_get", STATE_IDLE},
{MPITYPE_OTHER, 47, "MPI_Cart_map", STATE_IDLE},
{MPITYPE_OTHER, 48, "MPI_Cart_rank", STATE_IDLE},
{MPITYPE_OTHER, 44, "MPI_Cart_shift", STATE_IDLE},
{MPITYPE_OTHER, 49, "MPI_Cart_sub", STATE_IDLE},
{MPITYPE_OTHER, 50, "MPI_Cartdim_get", STATE_IDLE},
{MPITYPE_OTHER, 51, "MPI_Dims_create", STATE_IDLE},
{MPITYPE_OTHER, 52, "MPI_Graph_get", STATE_IDLE},
{MPITYPE_OTHER, 53, "MPI_Graph_map", STATE_IDLE},
{MPITYPE_OTHER, 55, "MPI_Graph_neighbors", STATE_IDLE},
{MPITYPE_OTHER, 54, "MPI_Graph_create", STATE_IDLE},
{MPITYPE_OTHER, 56, "MPI_Graphdims_get", STATE_IDLE},
{MPITYPE_OTHER, 57, "MPI_Graph_neighbors_count", STATE_IDLE},
{MPITYPE_OTHER, 58, "MPI_Topo_test", STATE_IDLE},
{MPITYPE_OTHER, 0, "End", STATE_IDLE},
// one sided
{MPITYPE_RMA, 63, "MPI_Win_create", STATE_IDLE},
{MPITYPE_RMA, 64, "MPI_Win_free", STATE_IDLE},
{MPITYPE_RMA, 65, "MPI_Put", STATE_IDLE},
{MPITYPE_RMA, 66, "MPI_Get", STATE_IDLE},
{MPITYPE_RMA, 67, "MPI_Accumulate", STATE_IDLE},
{MPITYPE_RMA, 68, "MPI_Win_fence", STATE_IDLE},
{MPITYPE_RMA, 69, "MPI_Win_complete", STATE_IDLE},
{MPITYPE_RMA, 70, "MPI_Win_start", STATE_IDLE},
{MPITYPE_RMA, 71, "MPI_Win_post", STATE_IDLE},
{MPITYPE_RMA, 72, "MPI_Win_wait", STATE_IDLE},
{MPITYPE_RMA, 73, "MPI_Win_test", STATE_IDLE},
{MPITYPE_RMA, 74, "MPI_Win_lock", STATE_IDLE},
{MPITYPE_RMA, 75, "MPI_Win_unlock", STATE_IDLE},
{MPITYPE_RMA, 0, "End", STATE_IDLE}
};

/* 
  This function is likely very slow.  It brute-forces through the
  list of MPI functions, and maps them to types and values.
  This should be replaced with a hash table or something like that.

  Map the name of the MPI function to the Paraver type and value.
  Also returns the state, but the state likely should just be "idle".
*/
int mapMPINameToTypeValue (char *name, int *type, int *value, int *state) {
  int i;
  char localName[128] = "";
  char *tmpPtr = strchr(name, 'M'); // trim the leading quote mark
  strcpy (localName, tmpPtr);
  int lastchar = strcspn(localName, "() ");
  localName[lastchar] = '\0';
  for (i = 0 ; i < NUM_MPI_FUNCTIONS ; i++) {
    if (strcmp(MPI_FUNCTION_MAP[i].name, localName) == 0) {
	  *type = MPI_FUNCTION_MAP[i].type;
	  *value = MPI_FUNCTION_MAP[i].value;
	  *state = MPI_FUNCTION_MAP[i].state;
	  return i;
	}
  }
  return -1;  // not found
}
