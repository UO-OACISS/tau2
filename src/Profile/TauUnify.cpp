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


#ifdef TAU_MPI
#ifdef TAU_EXP_UNIFY
#include <mpi.h>

#include <TauUtil.h>
#include <TauMetrics.h>
#include <Profiler.h>
#include <TauUnify.h>


typedef struct {
  vector<char*> strings;
  int *mapping;
  int numStrings;
} unify_merge_object_t;



// not the best style, but I use a global here to store the current event lister so that qsort can work
EventLister *theEventLister;


static int comparator(const void *p1, const void *p2) {
  int arg0 = *(int*)p1;
  int arg1 = *(int*)p2;
  return strcmp(theEventLister->getEvent(arg0),theEventLister->getEvent(arg1));
}


Tau_util_outputDevice *Tau_unify_generateLocalDefinitionBuffer(int *sortMap) {
  int numFuncs = theEventLister->getNumEvents();

  Tau_util_outputDevice *out = Tau_util_createBufferOutputDevice();

  Tau_util_output(out,"%d%c", numFuncs, '\0');
  for(int i=0;i<numFuncs;i++) {
    Tau_util_output(out,"%s%c", theEventLister->getEvent(i), '\0');
  }

  return out;
}

int *Tau_unify_generateSortMap() {
  int rank, numRanks, i;
  MPI_Status status;

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &numRanks);

  int numFuncs = theEventLister->getNumEvents();
  int *sortMap = (int*) TAU_UTIL_MALLOC(numFuncs*sizeof(int));

  for (int i=0; i<numFuncs; i++) {
    sortMap[i] = i;
  }

  qsort(sortMap, numFuncs, sizeof(int), comparator);

  return sortMap;
}


unify_object_t *Tau_unify_processBuffer(char *buffer, int rank) {
  // create the unification object
  unify_object_t *unifyObject = (unify_object_t*) malloc (sizeof(unify_object_t));
  unifyObject->buffer = buffer;
  unifyObject->rank = rank;

  int numFuncs;
  sscanf(buffer,"%d", &numFuncs);
  // printf ("Got %d funcs\n", numFuncs);
  char **strings = (char **) malloc(sizeof(char*) * numFuncs);
  buffer = strchr(buffer, '\0')+1;
  for (int i=0; i<numFuncs; i++) {
    strings[i] = buffer;
    // printf ("stringz[%d] = %s (%p)\n", i, strings[i], buffer);
    buffer = strchr(buffer, '\0')+1;
  }
  
  unifyObject->numFuncs = numFuncs;
  unifyObject->strings = strings;
  unifyObject->mapping = (int*) malloc (sizeof(int)*numFuncs);
  for (int i=0; i<numFuncs; i++) {
    unifyObject->mapping[i] = i;
  }
  return unifyObject;
}

Tau_util_outputDevice *Tau_unify_generateMergedDefinitionBuffer(unify_merge_object_t &mergedObject) {
  int numFuncs = theEventLister->getNumEvents();

  Tau_util_outputDevice *out = Tau_util_createBufferOutputDevice();

  Tau_util_output(out,"%d%c", mergedObject.strings.size(), '\0');
  for(int i=0;i<mergedObject.strings.size();i++) {
    Tau_util_output(out,"%s%c", mergedObject.strings[i], '\0');
  }

  return out;
}


unify_merge_object_t *Tau_unify_mergeObjects(vector<unify_object_t*> &objects) {
  unify_merge_object_t *mergedObject = new unify_merge_object_t();

  for (int i=0; i<objects.size(); i++) {
    // reset index pointers to start
    objects[i]->idx = 0;
  }

  bool finished = false;

  int count = 0;

  while (!finished) {
    // merge objects

    char *nextString = NULL;
    int objectIndex = 0;
    for (int i=0; i<objects.size(); i++) {
      if (objects[i]->idx < objects[i]->numFuncs) {
	if (nextString == NULL) {
	  nextString = objects[i]->strings[objects[i]->idx];
	  objectIndex = i;
	} else {
	  char *compareString = objects[i]->strings[objects[i]->idx];
	  if (strcmp(nextString, compareString) > 0) {
	    nextString = compareString;
	    objectIndex = i;
	  }
	}
      }
    }
 
    // the next string is given in nextString at this point
    mergedObject->strings.push_back(nextString);

    finished = true;

    // write the mappings and check if we are finished
    for (int i=0; i<objects.size(); i++) {
      if (objects[i]->idx < objects[i]->numFuncs) {
	char * compareString = objects[i]->strings[objects[i]->idx];
	if (strcmp(nextString, compareString) == 0) {
	  objects[i]->mapping[objects[i]->idx] = count;
	  objects[i]->idx++;
	}
	if (objects[i]->idx < objects[i]->numFuncs) {
	  finished = false;
	}
      }
    }

    count++;
    
  }

  mergedObject->numStrings = count;

  // for (int i=0; i<mergedObject->strings.size(); i++) {
  //   printf ("mergedObject->strings[%d] = %s\n", i, mergedObject->strings[i]);
  // }

  return mergedObject;
}




extern "C" int Tau_unify_unifyDefinitions() {
  FunctionEventLister *functionEventLister = new FunctionEventLister();
  Tau_unify_unifyEvents(functionEventLister);

  AtomicEventLister *atomicEventLister = new AtomicEventLister();
  Tau_unify_unifyEvents(atomicEventLister);

}



unify_object_t *Tau_unify_unifyEvents(EventLister *eventLister) {
  theEventLister = eventLister;
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


  // array of unifcation objects
  vector<unify_object_t*> *unifyObjects = new vector<unify_object_t*>();


  unify_merge_object_t *mergedObject = NULL;


  Tau_util_outputDevice *out = Tau_unify_generateLocalDefinitionBuffer(sortMap);

  // add ourselves
  char *defBuf = Tau_util_getOutputBuffer(out);
  int defBufSize = Tau_util_getOutputBufferLength(out);
  unifyObjects->push_back(Tau_unify_processBuffer(defBuf, -1));

  // use binomial heap algorithm (like MPI_Reduce)
  int mask = 0x1;
  int parent = -1;

  while (mask < numRanks) {
    if ((mask & rank) == 0) {
      int source = (rank | mask);
      if (source < numRanks) {
	
	/* send ok-to-go */
	PMPI_Send(NULL, 0, MPI_INT, source, 0, MPI_COMM_WORLD);
	
	/* receive buffer length */
	int recv_buflen;
	PMPI_Recv(&recv_buflen, 1, MPI_INT, source, 0, MPI_COMM_WORLD, &status);

	// allocate buffer
	char *recv_buf = (char *) TAU_UTIL_MALLOC(recv_buflen);

	/* receive buffer */
	PMPI_Recv(recv_buf, recv_buflen, MPI_CHAR, source, 0, MPI_COMM_WORLD, &status);

	// printf ("[%d] received from %d\n", rank, source);

	/* add unification object to array */
	unifyObjects->push_back(Tau_unify_processBuffer(recv_buf, source));
      }

    } else {
      /* I've received from all my children, now process and send the results up. */

      if (unifyObjects->size() > 1) {
	// merge children
	mergedObject = Tau_unify_mergeObjects(*unifyObjects);
	
	// generate buffer to send to parent
	Tau_util_outputDevice *out = Tau_unify_generateMergedDefinitionBuffer(*mergedObject);
	defBuf = Tau_util_getOutputBuffer(out);
	defBufSize = Tau_util_getOutputBufferLength(out);
      }

      parent = (rank & (~ mask));

      /* recieve ok to go */
      PMPI_Recv(NULL, 0, MPI_INT, parent, 0, MPI_COMM_WORLD, &status);
      
      /* send length */
      PMPI_Send(&defBufSize, 1, MPI_INT, parent, 0, MPI_COMM_WORLD);
      
      /* send data */
      PMPI_Send(defBuf, defBufSize, MPI_CHAR, parent, 0, MPI_COMM_WORLD);

      // printf ("[%d] sent to %d\n", rank, parent);
      break;
    }
    mask <<= 1;
  }


  if (rank == 0) {
    // rank 0 will now put together the final event id map
    mergedObject = Tau_unify_mergeObjects(*unifyObjects);

    for (int i=0; i<mergedObject->strings.size(); i++) {
      fprintf (stderr, "mergedObject->strings[%d] = %s\n", i, mergedObject->strings[i]);
    }

  }

  if (mergedObject == NULL) {
    // leaf functions allocate a phony merged object to use below
    int numFuncs = theEventLister->getNumEvents();
    mergedObject = new unify_merge_object_t();
    mergedObject->numStrings = numFuncs;
  }

  // receive back table from parent
  if (parent != -1) {
    // printf ("allocating %d items\n", mergedObject->numStrings);
    mergedObject->mapping = (int *) TAU_UTIL_MALLOC(sizeof(int)* mergedObject->numStrings);
    
    // printf ("Receiving from %d\n", parent);
    PMPI_Recv(mergedObject->mapping, mergedObject->numStrings, MPI_INT, parent, 0, MPI_COMM_WORLD, &status);

    // apply mapping table to children
    for (int i=0; i<unifyObjects->size(); i++) {
      for (int j=0; j<(*unifyObjects)[i]->numFuncs; j++) {
	(*unifyObjects)[i]->mapping[j] = mergedObject->mapping[(*unifyObjects)[i]->mapping[j]];
      }
    }
  }

  // send tables to children
  for (int i=1; i<unifyObjects->size(); i++) {
    // printf ("Sending to %d\n", (*unifyObjects)[i]->rank);
    PMPI_Send((*unifyObjects)[i]->mapping, (*unifyObjects)[i]->numFuncs, MPI_INT, (*unifyObjects)[i]->rank, 0, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    unify_object_t *object = (*unifyObjects)[0];
    for (int i=0; i<object->numFuncs; i++) {
      fprintf (stderr, "[rank %d] = Entry %d maps to [%d] is %s\n", rank, i, object->mapping[i], object->strings[i]);
    }
  }

  if (rank == 0) {
    end = TauMetrics_getTimeOfDay();
    eventLister->setDuration(((double)(end-start))/1000000.0f);
    TAU_VERBOSE("TAU: Unifying Complete, duration = %.4G seconds\n", ((double)(end-start))/1000000.0f);
    char tmpstr[256];
    sprintf(tmpstr, "%.4G seconds", ((double)(end-start))/1000000.0f);
    TAU_METADATA("TAU Unification Time", tmpstr);
  }


  unify_object_t *object = (*unifyObjects)[0];
  return object;
}


#endif /* TAU_EXP_UNIFY */
#endif /* TAU_MPI */
