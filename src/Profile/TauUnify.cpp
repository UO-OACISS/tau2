/****************************************************************************
**			TAU Portable Profiling Package			   **
**			http://www.cs.uoregon.edu/research/tau	           **
*****************************************************************************
**    Copyright 2010                                                       **
**    Department of Computer and Information Science, University of Oregon **
**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
****************************************************************************/
/****************************************************************************
**	File            : TauUnify.cpp                                     **
**	Contact		: tau-bugs@cs.uoregon.edu                          **
**	Documentation	: See http://tau.uoregon.edu                       **
**                                                                         **
**      Description     : Event unification                                **
**                                                                         **
****************************************************************************/


// int *local_id_map;
// int *

#ifdef TAU_MPI
#include <mpi.h>

#include <mpi.h>
#include <TauUtil.h>
#include <TauMetrics.h>
#include <Profiler.h>

typedef struct {
  int numFuncs;
  char **strings;
  int *mapping;
  int idx;
} unify_object_t;




static int comparator(const void *p1, const void *p2) {
  int arg0 = *(int*)p1;
  int arg1 = *(int*)p2;
  return strcmp(TheFunctionDB()[arg0]->GetName(),TheFunctionDB()[arg1]->GetName());
}


Tau_util_outputDevice *Tau_unify_generateLocalDefinitionBuffer(int *sortMap) {
  int numFuncs = TheFunctionDB().size();

  Tau_util_outputDevice *out = Tau_util_createBufferOutputDevice();
  if (out == NULL) {
    TAU_ABORT("TAU: Abort: Unable to generate create buffer for local definitions\n");
  }

  Tau_util_output(out,"%d%c", numFuncs, '\0');
  for(int i=0;i<numFuncs;i++) {
    FunctionInfo *fi = TheFunctionDB()[sortMap[i]];
    Tau_util_output(out,"%s%c", fi->GetName(), '\0');
  }

  return out;
}

int *Tau_unify_generateSortMap() {
  int rank, numRanks, i;
  MPI_Status status;

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  int numFuncs = TheFunctionDB().size();
  int *sortMap = (int*) malloc(numFuncs*sizeof(int));
  if (sortMap == NULL) {
    TAU_ABORT("TAU: Abort: Unable to allocate memory\n");
  }
  for (int i=0; i<numFuncs; i++) {
    sortMap[i] = i;
  }

  qsort(sortMap, numFuncs, sizeof(int), comparator);

  for (int i=0; i<numFuncs; i++) {
    printf ("[%d] sortMap[%d] = %d (%s)\n", rank, i, sortMap[i], TheFunctionDB()[sortMap[i]]->GetName());
  }
  return sortMap;
}


unify_object_t *Tau_unify_processBuffer(char *buffer) {
  int numFuncs;
  sscanf(buffer,"%d", &numFuncs);
  printf ("Got %d funcs\n", numFuncs);
  char **strings = (char **) malloc(sizeof(char*) * numFuncs);
  buffer = strchr(buffer, '\0')+1;
  for (int i=0; i<numFuncs; i++) {
    strings[i] = buffer;
    printf ("strings[%d] = %s (%p)\n", i, strings[i], buffer);
    buffer = strchr(buffer, '\0')+1;
  }
  
  // create the unification object
  unify_object_t *unifyObject = (unify_object_t*) malloc (sizeof(unify_object_t));
  unifyObject->numFuncs = numFuncs;
  unifyObject->strings = strings;
  unifyObject->mapping = (int*) malloc (sizeof(int)*numFuncs);
  return unifyObject;
}


void Tau_unify_mergeObjects(vector<unify_object_t*> objects) {
  unify_object_t *mergedObject = (unify_object_t*) malloc (sizeof(unify_object_t));
  
  for (int i=0; i<objects.size(); i++) {
    objects[i]->idx = 0;
  }

  bool finished = false;

  vector<char*> newStrings;

  while (!finished) {
    // merge objects

    char *nextString = objects[0]->strings[objects[0]->idx];
    int objectIndex = 0;
    for (int i=1; i<objects.size(); i++) {
      char * compareString = objects[i]->strings[objects[i]->idx];
      if (strcmp(nextString, compareString) < 0) {
	nextString = compareString;
	objectIndex = i;
      }
    }
 
    // The next string is given in nextString at this point


    finished = true;

  }

}

extern "C" int Tau_unify_unifyDefinitions() {
  int rank, numRanks, i;
  MPI_Status status;

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &numRanks);
  x_uint64 start, end;

  if (rank == 0) {
    TAU_VERBOSE("TAU: Unifying...\n");
    start = TauMetrics_getTimeOfDay();
  }

  int *sortMap = Tau_unify_generateSortMap();
  Tau_util_outputDevice *out = Tau_unify_generateLocalDefinitionBuffer(sortMap);

  if (!out) {
    TAU_ABORT("TAU: Abort: Unable to generate local definitions\n");
  }

  char *defBuf = Tau_util_getOutputBuffer(out);
  int defBufSize = Tau_util_getOutputBufferLength(out);

  TAU_VERBOSE("UNIFY: [%d] - My def buf size = %d\n", rank, defBufSize);

  // determine maximum buffer size
  int maxDefBufSize;
  PMPI_Allreduce(&defBufSize, &maxDefBufSize, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  
  // allocate receive buffer
  char *recv_buf = (char *) malloc (maxDefBufSize);
  if (recv_buf == NULL) {
    TAU_ABORT("TAU: Abort: Unable to allocate recieve buffer for unification\n");
  }

  // use binomial heap algorithm (like MPI_Reduce)
  int mask = 0x1;


  // array of unifcation objects
  vector<unify_object_t*> unifyObjects;

  // add ourselves
  unifyObjects.push_back(Tau_unify_processBuffer(defBuf));

  while (mask < numRanks) {
    if ((mask & rank) == 0) {
      int source = (rank | mask);
      if (source < numRanks) {
	
	/* send ok-to-go */
	PMPI_Send(NULL, 0, MPI_INT, source, 0, MPI_COMM_WORLD);
	
	/* receive buffer length */
	int recv_buflen;
	PMPI_Recv(&recv_buflen, 1, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
	
	/* receive buffer */
	PMPI_Recv(recv_buf, recv_buflen, MPI_CHAR, source, 0, MPI_COMM_WORLD, &status);

	printf ("[%d] received from %d\n", rank, source);

	/* add unification object to array */
	unifyObjects.push_back(Tau_unify_processBuffer(recv_buf));
      }

    } else {
      /* I've received from all my children, now process and send the results up. */

      Tau_unify_mergeObjects(unifyObjects);


      int target = (rank & (~ mask));

      /* recieve ok to go */
      PMPI_Recv(NULL, 0, MPI_INT, target, 0, MPI_COMM_WORLD, &status);
      
      /* send length */
      PMPI_Send(&defBufSize, 1, MPI_INT, target, 0, MPI_COMM_WORLD);
      
      /* send data */
      PMPI_Send(defBuf, defBufSize, MPI_CHAR, target, 0, MPI_COMM_WORLD);

      printf ("[%d] sent to %d\n", rank, target);
      break;
    }
    mask <<= 1;
  }


  free (recv_buf);

  if (rank == 0) {
    end = TauMetrics_getTimeOfDay();
    TAU_VERBOSE("TAU: Unifying Complete, duration = %.4G seconds\n", ((double)(end-start))/1000000.0f);
  }

  return 0;
}



#endif /* TAU_MPI */
