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


//#ifdef TAU_MPI

#if (defined(TAU_MPI) || defined (TAU_MPC))
#include <mpi.h>
#endif /* TAU_MPI */

#ifdef TAU_SHMEM
#include <shmem.h>
extern "C" void  __real_shmem_int_put(int * a1, const int * a2, size_t a3, int a4) ;
extern "C" void  __real_shmem_int_get(int * a1, const int * a2, size_t a3, int a4) ;
extern "C" void  __real_shmem_putmem(void * a1, const void * a2, size_t a3, int a4) ;
extern "C" void  __real_shmem_barrier_all() ;
extern "C" void  __real_shmem_quiet() ;
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
extern "C" int   __real__num_pes() ;
extern "C" int   __real__my_pe() ;
extern "C" void* __real_shmalloc(size_t a1) ;
extern "C" void  __real_shfree(void * a1) ;
#else
extern "C" int   __real_shmem_n_pes() ;
extern "C" int   __real_shmem_my_pe() ;
extern "C" void* __real_shmem_malloc(size_t a1) ;
extern "C" void  __real_shmem_free(void * a1) ;
#endif /* SHMEM_1_1 || SHMEM_1_2 */
#endif /* TAU_SHMEM */

//#ifdef TAU_UNIFY

#include <TauUtil.h>
#include <TauMetrics.h>
#include <Profiler.h>
#include <TauUnify.h>

#include <algorithm>
using namespace std;

#ifdef TAU_UNIFY
/** local unification object, one is created for each child rank that we talk to */
typedef struct {
  int rank;       // MPI rank of child
  char *buffer;   // buffer given to us by rank
  int numEvents;  // number of events
  char **strings; // pointers into buffer for strings
  int *mapping;   // mapping table for this child
  int idx;        // index used for merge operation
  int *sortMap;   // sort map for this rank
  int globalNumItems;  // global number of items
} unify_object_t;

/** unification merge object */
typedef struct {

  /** This is a vector of pointers to currently existing strings
   *  inside the contiguous buffers from child ranks */
  vector<char*> strings;

  /* the number of entries, we can't use strings.size() because the merged
     strings only exist on the parent */
  int numStrings;

  /* mapping table */
  int *mapping;

} unify_merge_object_t;



/** Comparator class used to create a sort map for unification */
class EventComparator : public binary_function<int, int, bool> {

private:
  EventLister *eventLister;

public:

  /** Constructor takes an EventLister, stores it for use with comparison */
  EventComparator(EventLister *eventLister) {
    this->eventLister = eventLister;
  }

  /** Compare two integers based on the strings that they index */
  bool operator() (int l1, int l2) const {
    return strcmp(eventLister->getEvent(l1),eventLister->getEvent(l2)) < 0;
  }
};


/** Return a table represeting a sorted list of the events */
int *Tau_unify_generateSortMap_MPI(EventLister *eventLister) {
#ifdef TAU_MPI
  int rank = 0;
  int numRanks = 1;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &numRanks);
#endif /* TAU_MPI */

  int numEvents = eventLister->getNumEvents();
  int *sortMap = (int*) TAU_UTIL_MALLOC(numEvents*sizeof(int));

  for (int i=0; i<numEvents; i++) {
    sortMap[i] = i;
  }

  sort(sortMap, sortMap + numEvents, EventComparator(eventLister));

  return sortMap;
}

int *Tau_unify_generateSortMap_SHMEM(EventLister *eventLister) {
#ifdef TAU_SHMEM
  int rank = 0;
  int numRanks = 1;
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  numRanks = __real__num_pes();
  rank = __real__my_pe();
#else
  numRanks = __real_shmem_n_pes();
  rank = __real_shmem_my_pe();
#endif /* SHMEM_1_1 || SHMEM_1_2 */
#endif /* TAU_SHMEM */

  int numEvents = eventLister->getNumEvents();
  int *sortMap = (int*) TAU_UTIL_MALLOC(numEvents*sizeof(int));

  for (int i=0; i<numEvents; i++) {
    sortMap[i] = i;
  }

  sort(sortMap, sortMap + numEvents, EventComparator(eventLister));

  return sortMap;
}


/** Return a Tau_util_outputDevice containing a buffer of the event definitions */
Tau_util_outputDevice *Tau_unify_generateLocalDefinitionBuffer(int *sortMap, EventLister *eventLister) {
  int numEvents = eventLister->getNumEvents();

  // create a buffer-based output device
  Tau_util_outputDevice *out = Tau_util_createBufferOutputDevice();

  // write the number of events into the output device
  Tau_util_output(out,"%d%c", numEvents, '\0');

  // write each event into the output device
  for(int i=0;i<numEvents;i++) {
    Tau_util_output(out,"%s%c", eventLister->getEvent(sortMap[i]), '\0');
  }

  return out;
}

/** Process a buffer from a given rank, return a unify_object_t */
unify_object_t *Tau_unify_processBuffer(char *buffer, int rank) {

  // create the unification object
  unify_object_t *unifyObject = (unify_object_t*) TAU_UTIL_MALLOC(sizeof(unify_object_t));
  unifyObject->buffer = buffer;
  unifyObject->rank = rank;

  // read the number of events from the buffer
  int numEvents;
  sscanf(buffer,"%d", &numEvents);
  unifyObject->numEvents = numEvents;

  // assign the "string" pointers to their locations within the buffer
  unifyObject->strings = (char **) TAU_UTIL_MALLOC(sizeof(char*) * numEvents);
  buffer = strchr(buffer, '\0')+1;
  for (int i=0; i<numEvents; i++) {
    unifyObject->strings[i] = buffer;
    buffer = strchr(buffer, '\0')+1;
  }

  // create an initial mapping table for this rank
  unifyObject->mapping = (int*) TAU_UTIL_MALLOC(sizeof(int) * numEvents);
  for (int i=0; i<numEvents; i++) {
    unifyObject->mapping[i] = i;
  }
  return unifyObject;
}

/** Generates a definition buffer from a unify_merge_object_t */
Tau_util_outputDevice *Tau_unify_generateMergedDefinitionBuffer(unify_merge_object_t &mergedObject,
								EventLister *eventLister) {
  Tau_util_outputDevice *out = Tau_util_createBufferOutputDevice();

  Tau_util_output(out,"%d%c", mergedObject.strings.size(), '\0');
  for(unsigned int i=0;i<mergedObject.strings.size();i++) {
    Tau_util_output(out,"%s%c", mergedObject.strings[i], '\0');
  }

  return out;
}

/** Merge a set of unification objects.  Because each set of event identifiers is sorted,
    this is a simple merge operation. */
unify_merge_object_t *Tau_unify_mergeObjects(vector<unify_object_t*> &objects) {
  unify_merge_object_t *mergedObject = new unify_merge_object_t();

  for (unsigned int i=0; i<objects.size(); i++) {
    // reset index pointers to start
    objects[i]->idx = 0;
  }

  bool finished = false;

  int count = 0;

  while (!finished) {
    // merge objects

    char *nextString = NULL;
    for (unsigned int i=0; i<objects.size(); i++) {
      if (objects[i]->idx < objects[i]->numEvents) {
        if (nextString == NULL) {
          nextString = objects[i]->strings[objects[i]->idx];
        } else {
          char *compareString = objects[i]->strings[objects[i]->idx];
          if (strcmp(nextString, compareString) > 0) {
            nextString = compareString;
          }
        }
      }
    }

    // the next string is given in nextString at this point
    if (nextString != NULL) {
      mergedObject->strings.push_back(nextString);
    }

    finished = true;

    // write the mappings and check if we are finished
    for (unsigned int i=0; i<objects.size(); i++) {
      if (objects[i]->idx < objects[i]->numEvents) {
	char * compareString = objects[i]->strings[objects[i]->idx];
	if (strcmp(nextString, compareString) == 0) {
	  objects[i]->mapping[objects[i]->idx] = count;
	  objects[i]->idx++;
	}
	if (objects[i]->idx < objects[i]->numEvents) {
	  finished = false;
	}
      }
    }

    if (nextString != NULL) {
      count++;
    }
  }

  mergedObject->numStrings = count;

  return mergedObject;
}



/** Using MPI, unify events for a given EventLister */
Tau_unify_object_t *Tau_unify_unifyEvents_MPI(EventLister *eventLister) {
  int rank, numRanks;
  rank = 0;
  numRanks = 1;
#ifdef TAU_MPI
  MPI_Status status;

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &numRanks);
#endif /* TAU_MPI */

  // for internal timing
  x_uint64 start, end;

  if (rank == 0) {
    TAU_VERBOSE("TAU: Unifying...\n");
    start = TauMetrics_getTimeOfDay();
  }

  // generate our own sort map
  int *sortMap = Tau_unify_generateSortMap_MPI(eventLister);

  // array of unification objects
  vector<unify_object_t*> *unifyObjects = new vector<unify_object_t*>();

  // add ourselves
  Tau_util_outputDevice *out = Tau_unify_generateLocalDefinitionBuffer(sortMap, eventLister);
  char *defBuf = Tau_util_getOutputBuffer(out);
  int defBufSize = Tau_util_getOutputBufferLength(out);
  unifyObjects->push_back(Tau_unify_processBuffer(defBuf, -1 /* no rank */));


  // define our merge object
  unify_merge_object_t *mergedObject = NULL;

  // use binomial heap (like MPI_Reduce) to communicate with parent/children
  int mask = 0x1;
  int parent = -1;

  while (mask < numRanks) {
    if ((mask & rank) == 0) {
      int source = (rank | mask);
      if (source < numRanks) {

	int recv_buflen = 0;

#ifdef TAU_MPI
	// send ok-to-go
	PMPI_Send(NULL, 0, MPI_INT, source, 0, MPI_COMM_WORLD);

	// receive buffer length
	PMPI_Recv(&recv_buflen, 1, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
#endif /* TAU_MPI */

	// Only receive and allocate memory if there's something to receive.
	//   Note that this condition only applies to Atomic events.
	if (recv_buflen > 0) {
	  // allocate buffer
	  char *recv_buf = (char *) TAU_UTIL_MALLOC(recv_buflen);

#ifdef TAU_MPI
	  // receive buffer
	  PMPI_Recv(recv_buf, recv_buflen, MPI_CHAR, source, 0, MPI_COMM_WORLD, &status);
#endif /* TAU_MPI */

	  // add unification object to array
	  unifyObjects->push_back(Tau_unify_processBuffer(recv_buf, source));
	}
      }

    } else {
      // I've received from all my children, now process and send the results up.

      if (unifyObjects->size() > 1) {
	// merge children
	mergedObject = Tau_unify_mergeObjects(*unifyObjects);

	// generate buffer to send to parent
	Tau_util_outputDevice *out = Tau_unify_generateMergedDefinitionBuffer(*mergedObject, eventLister);
	defBuf = Tau_util_getOutputBuffer(out);
	defBufSize = Tau_util_getOutputBufferLength(out);
      }

      parent = (rank & (~ mask));

#ifdef TAU_MPI
      // recieve ok to go
      PMPI_Recv(NULL, 0, MPI_INT, parent, 0, MPI_COMM_WORLD, &status);

      // send length
      PMPI_Send(&defBufSize, 1, MPI_INT, parent, 0, MPI_COMM_WORLD);
#endif /* TAU_MPI */

      // Send data only if the buffer size is greater than 0.
      //   This applies only to Atomic events.
      if (defBufSize > 0) {
#ifdef TAU_MPI
	// send data
	PMPI_Send(defBuf, defBufSize, MPI_CHAR, parent, 0, MPI_COMM_WORLD);
#endif /* TAU_MPI */
      }
      break;
    }
    mask <<= 1;
  }

  int globalNumItems;

  if (rank == 0) {
    // rank 0 will now put together the final event id map
    mergedObject = Tau_unify_mergeObjects(*unifyObjects);

    globalNumItems = mergedObject->strings.size();
  }


  if (mergedObject == NULL) {
    // leaf functions allocate a phony merged object to use below
    int numEvents = eventLister->getNumEvents();
    mergedObject = new unify_merge_object_t();
    mergedObject->numStrings = numEvents;
  }

  // receive reverse mapping table from parent
  if (parent != -1) {
    mergedObject->mapping = (int *) TAU_UTIL_MALLOC(sizeof(int)* mergedObject->numStrings);

#ifdef TAU_MPI
    PMPI_Recv(mergedObject->mapping, mergedObject->numStrings,
	      MPI_INT, parent, 0, MPI_COMM_WORLD, &status);
#endif /* TAU_MPI */

    // apply mapping table to children
    for (unsigned int i=0; i<unifyObjects->size(); i++) {
      for (int j=0; j<(*unifyObjects)[i]->numEvents; j++) {
	(*unifyObjects)[i]->mapping[j] = mergedObject->mapping[(*unifyObjects)[i]->mapping[j]];
      }
    }
  }

  // send tables to children
  for (unsigned int i=1; i<unifyObjects->size(); i++) {
#ifdef TAU_MPI
    PMPI_Send((*unifyObjects)[i]->mapping, (*unifyObjects)[i]->numEvents,
	      MPI_INT, (*unifyObjects)[i]->rank, 0, MPI_COMM_WORLD);
#endif /* TAU_MPI */
  }

  /* debug: output final table */
  // if (rank == 0) {
  //   unify_object_t *object = (*unifyObjects)[0];
  //   for (int i=0; i<object->numEvents; i++) {
  //     fprintf (stderr, "[rank %d] = Entry %d maps to [%d] is %s\n", rank, i, object->mapping[i], object->strings[i]);
  //   }
  // }

  if (rank == 0) {
    // finalize timing and write into metadata
    end = TauMetrics_getTimeOfDay();
    eventLister->setDuration(((double)(end-start))/1000000.0f);
    TAU_VERBOSE("TAU: Unifying Complete, duration = %.4G seconds\n",
		((double)(end-start))/1000000.0f);
    char tmpstr[256];
    snprintf(tmpstr, sizeof(tmpstr),  "%.4G seconds", ((double)(end-start))/1000000.0f);
    TAU_METADATA("TAU Unification Time", tmpstr);
  }

  // the local object
  unify_object_t *object = (*unifyObjects)[0];

#ifdef TAU_MPI
  PMPI_Bcast (&globalNumItems, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif /* TAU_MPI */

  Tau_unify_object_t *tau_unify_object = (Tau_unify_object_t*) TAU_UTIL_MALLOC(sizeof(Tau_unify_object_t));
  tau_unify_object->globalNumItems = globalNumItems;
  tau_unify_object->sortMap = sortMap;
  tau_unify_object->mapping = object->mapping;
  tau_unify_object->localNumItems = object->numEvents;
  tau_unify_object->globalStrings = NULL;

  if (rank == 0) {
    char **globalStrings = (char**)TAU_UTIL_MALLOC(sizeof(char*)*globalNumItems);

    for (unsigned int i=0; i<mergedObject->strings.size(); i++) {
      globalStrings[i] = strdup(mergedObject->strings[i]);
    }
    tau_unify_object->globalStrings = globalStrings;
  }

  /* free up memory */
  delete mergedObject;

  Tau_util_destroyOutputDevice(out);

  free ((*unifyObjects)[0]->strings);
  free ((*unifyObjects)[0]);

  for (unsigned int i=1; i<unifyObjects->size(); i++) {
    //free ((*unifyObjects)[i]->buffer);
    free ((*unifyObjects)[i]->strings);
    free ((*unifyObjects)[i]->mapping);
    free ((*unifyObjects)[i]);
  }
  delete unifyObjects;

  // return the unification object that will be used to map local <-> global ids
  return tau_unify_object;
}

/** Using SHMEM, unify events for a given EventLister */
Tau_unify_object_t *Tau_unify_unifyEvents_SHMEM(EventLister *eventLister) {
  int rank = 0;
#ifdef TAU_SHMEM
  int numRanks = 1;
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  rank = __real__my_pe();
  numRanks = __real__num_pes();
#else
  rank = __real_shmem_my_pe();
  numRanks = __real_shmem_n_pes();
#endif /* SHMEM_1_1 || SHMEM_1_2 */
#endif /* TAU_SHMEM */

  // for internal timing
  x_uint64 start, end;

  if (rank == 0) {
    TAU_VERBOSE("TAU: Unifying...\n");
    start = TauMetrics_getTimeOfDay();
  }

  // generate our own sort map
  int *sortMap = Tau_unify_generateSortMap_SHMEM(eventLister);

  // array of unification objects
  vector<unify_object_t*> *unifyObjects = new vector<unify_object_t*>();

  // add ourselves
  Tau_util_outputDevice *out = Tau_unify_generateLocalDefinitionBuffer(sortMap, eventLister);
  char *defBuf = Tau_util_getOutputBuffer(out);
  int defBufSize = Tau_util_getOutputBufferLength(out);
  TAU_UNUSED(defBufSize); // set again later, but fixing compiler warning
  unifyObjects->push_back(Tau_unify_processBuffer(defBuf, -1 /* no rank */));


  // define our merge object
  unify_merge_object_t *mergedObject = NULL;

#ifdef TAU_SHMEM
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  int *shmaxbuf = (int*)__real_shmalloc(sizeof(int));
  int *shmaxbufArr = (int*)__real_shmalloc(numRanks*sizeof(int));
#else
  int *shmaxbuf = (int*)__real_shmem_malloc(sizeof(int));
  int *shmaxbufArr = (int*)__real_shmem_malloc(numRanks*sizeof(int));
#endif /* SHMEM_1_1 || SHMEM_1_2 */

  char *shbuff;
  int i;


  int mask = 0x1;
  int parent = -1;
  int source;
  int break_flag = 0;
  while (mask < numRanks) {
    source = (rank | mask);
    for(i=0; i<numRanks;i++) shmaxbufArr[i] = 0;

    // sender
    if ((mask & rank) != 0) {
      if (unifyObjects->size() > 1) {
	// merge children
	mergedObject = Tau_unify_mergeObjects(*unifyObjects);

	// generate buffer to send to parent
	Tau_util_outputDevice *out = Tau_unify_generateMergedDefinitionBuffer(*mergedObject, eventLister);
	defBuf = Tau_util_getOutputBuffer(out);
	defBufSize = Tau_util_getOutputBufferLength(out);
      }

      // Send all defBufSize's to rank 0.
      __real_shmem_int_put(&shmaxbufArr[rank], &defBufSize, 1, 0);
    }
    __real_shmem_barrier_all();
    // Compute max buffer size on rank 0 and send to all pes.
    if (rank == 0) {
      *shmaxbuf = 0;
      for(i = 0; i < numRanks; i++) {
        if(shmaxbufArr[i] > *shmaxbuf) *shmaxbuf= shmaxbufArr[i];
      }
    }
    __real_shmem_barrier_all();
    __real_shmem_int_get(shmaxbuf, shmaxbuf, 1, 0);

#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
    shbuff = (char*)__real_shmalloc(*shmaxbuf);
#else
    shbuff = (char*)__real_shmem_malloc(*shmaxbuf);
#endif /* SHMEM_1_1 || SHMEM_1_2 */

    // sender
    if((mask & rank) != 0 && !break_flag) {
      parent = (rank & (~ mask));
      __real_shmem_putmem(shbuff, defBuf, defBufSize, parent);
    }
    __real_shmem_barrier_all();

    // receiver
    if((mask & rank) == 0 && source < numRanks && !break_flag) {
       unifyObjects->push_back(Tau_unify_processBuffer(shbuff, source));
    }
    else {
       break_flag = 1;
    }
    __real_shmem_barrier_all();

    mask <<= 1;
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
    __real_shfree(shbuff);
#else
    __real_shmem_free(shbuff);
#endif /* SHMEM_1_1 || SHMEM_1_2 */
  }
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  __real_shfree(shmaxbuf);
  __real_shfree(shmaxbufArr);
#else
  __real_shmem_free(shmaxbuf);
  __real_shmem_free(shmaxbufArr);
#endif /* SHMEM_1_1 || SHMEM_1_2 */
  __real_shmem_barrier_all();

#endif /* TAU_SHMEM */

  int globalNumItems = 0;

  if (rank == 0) {
    // rank 0 will now put together the final event id map
    mergedObject = Tau_unify_mergeObjects(*unifyObjects);

    globalNumItems = mergedObject->strings.size();
  }


  if (mergedObject == NULL) {
    // leaf functions allocate a phony merged object to use below
    int numEvents = eventLister->getNumEvents();
    mergedObject = new unify_merge_object_t();
    mergedObject->numStrings = numEvents;
  }

  // receive reverse mapping table from parent
#ifdef TAU_SHMEM
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  int *shreceived_mapping = (int*)__real_shmalloc(sizeof(int));
  int *shmergedObject_mapping = (int*)__real_shmalloc(mergedObject->numStrings*sizeof(int));
#else
  int *shreceived_mapping = (int*)__real_shmem_malloc(sizeof(int));
  int *shmergedObject_mapping = (int*)__real_shmem_malloc(mergedObject->numStrings*sizeof(int));
#endif /* SHMEM_1_1 || SHMEM_1_2 */
  int sent=1;
  *shreceived_mapping = 0;
  for(i=0;i<mergedObject->numStrings;i++) shmergedObject_mapping[i] = -69;
  while(parent != -1 && *shreceived_mapping == 0) {
    sleep(0);
  }
  __real_shmem_quiet();
  if (parent != -1) {
    for (i=0; i<unifyObjects->size(); i++) {
      for (int j=0; j<(*unifyObjects)[i]->numEvents; j++) {
        (*unifyObjects)[i]->mapping[j] = shmergedObject_mapping[(*unifyObjects)[i]->mapping[j]];
      }
    }
  }
  for (unsigned int i=1; i<unifyObjects->size(); i++) {
      __real_shmem_int_put(shmergedObject_mapping, (*unifyObjects)[i]->mapping, (*unifyObjects)[i]->numEvents, (*unifyObjects)[i]->rank);
      __real_shmem_int_put(shreceived_mapping, &sent, 1, (*unifyObjects)[i]->rank);
  }
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  __real_shfree(shreceived_mapping);
  __real_shfree(shmergedObject_mapping);
#else
  __real_shmem_free(shreceived_mapping);
  __real_shmem_free(shmergedObject_mapping);
#endif /* SHMEM_1_1 || SHMEM_1_2 */
#endif

  /* debug: output final table */
  // if (rank == 0) {
  //   unify_object_t *object = (*unifyObjects)[0];
  //   for (int i=0; i<object->numEvents; i++) {
  //     fprintf (stderr, "[rank %d] = Entry %d maps to [%d] is %s\n", rank, i, object->mapping[i], object->strings[i]);
  //   }
  // }

  if (rank == 0) {
    // finalize timing and write into metadata
    end = TauMetrics_getTimeOfDay();
    eventLister->setDuration(((double)(end-start))/1000000.0f);
    TAU_VERBOSE("TAU: Unifying Complete, duration = %.4G seconds\n",
		((double)(end-start))/1000000.0f);
    char tmpstr[256];
    snprintf(tmpstr, sizeof(tmpstr),  "%.4G seconds", ((double)(end-start))/1000000.0f);
    TAU_METADATA("TAU Unification Time", tmpstr);
  }

  // the local object
  unify_object_t *object = (*unifyObjects)[0];

#ifdef TAU_SHMEM
#if defined(SHMEM_1_1) || defined(SHMEM_1_2)
  int *shglobalNumItems = (int*)__real_shmalloc(sizeof(int));
  *shglobalNumItems = globalNumItems;
  __real_shmem_barrier_all();
  __real_shmem_int_get(&globalNumItems, shglobalNumItems, 1, 0);
  __real_shmem_barrier_all();
  __real_shfree(shglobalNumItems);
#else
  int *shglobalNumItems = (int*)__real_shmem_malloc(sizeof(int));
  *shglobalNumItems = globalNumItems;
  __real_shmem_barrier_all();
  __real_shmem_int_get(&globalNumItems, shglobalNumItems, 1, 0);
  __real_shmem_barrier_all();
  __real_shmem_free(shglobalNumItems);
#endif /* SHMEM_1_1 || SHMEM_1_2 */
#endif /* TAU_SHMEM */

  Tau_unify_object_t *tau_unify_object = (Tau_unify_object_t*) TAU_UTIL_MALLOC(sizeof(Tau_unify_object_t));
  tau_unify_object->globalNumItems = globalNumItems;
  tau_unify_object->sortMap = sortMap;
  tau_unify_object->mapping = object->mapping;
  tau_unify_object->localNumItems = object->numEvents;
  tau_unify_object->globalStrings = NULL;

  if (rank == 0) {
    char **globalStrings = (char**)TAU_UTIL_MALLOC(sizeof(char*)*globalNumItems);

    for (unsigned int i=0; i<mergedObject->strings.size(); i++) {
      globalStrings[i] = strdup(mergedObject->strings[i]);
    }
    tau_unify_object->globalStrings = globalStrings;
  }

  /* free up memory */
  delete mergedObject;

  Tau_util_destroyOutputDevice(out);

  free ((*unifyObjects)[0]->strings);
  free ((*unifyObjects)[0]);

  for (unsigned int i=1; i<unifyObjects->size(); i++) {
    //free ((*unifyObjects)[i]->buffer);
    free ((*unifyObjects)[i]->strings);
    free ((*unifyObjects)[i]->mapping);
    free ((*unifyObjects)[i]);
  }
  delete unifyObjects;

  // return the unification object that will be used to map local <-> global ids
  return tau_unify_object;
}

/** We store a unifier for the functions and atomic events for use externally */
/* *CWL* 2010-10-11: Is this safe? Are threads not used here? */
Tau_unify_object_t *functionUnifier=0, *atomicUnifier=0;
extern "C" Tau_unify_object_t *Tau_unify_getFunctionUnifier() {
  return functionUnifier;
}
extern "C" Tau_unify_object_t *Tau_unify_getAtomicUnifier() {
  return atomicUnifier;
}

/** Merge both function and atomic event definitions */
extern "C" int Tau_unify_unifyDefinitions_MPI() {
  FunctionEventLister *functionEventLister = new FunctionEventLister();
  functionUnifier = Tau_unify_unifyEvents_MPI(functionEventLister);
  AtomicEventLister *atomicEventLister = new AtomicEventLister();
  atomicUnifier = Tau_unify_unifyEvents_MPI(atomicEventLister);
  return 0;
}

extern "C" int Tau_unify_unifyDefinitions_SHMEM() {
  FunctionEventLister *functionEventLister = new FunctionEventLister();
  functionUnifier = Tau_unify_unifyEvents_SHMEM(functionEventLister);
  AtomicEventLister *atomicEventLister = new AtomicEventLister();
  atomicUnifier = Tau_unify_unifyEvents_SHMEM(atomicEventLister);
  return 0;
}


#endif /* TAU_UNIFY */

#ifdef TAU_MPC
/*extern "C" int TauInitMpcThreads(int* rank) {
  static bool firsttime = true;
  if (firsttime) {
    for (int i = 0; i < TAU_MAX_THREADS; i++) {
      rank[i] = -1;
    }
    firsttime = false;
  }
  return 0;
}*/
struct RankList : vector<int>{ //TODO: DYNATHREAD Test this implementation with a working MPC + merged profile output
 virtual ~RankList(){
         //printf("Destroying RankList at %p, with size %ld\n", this, this->size());
         Tau_destructor_trigger();
     }
};

extern "C" int TauGetMpiRank(void) {
  //static int firsttime = 1;
  static RankList rank;//int *rank = NULL;
  int retval;

  RtsLayer::LockDB();
  int tid = RtsLayer::myThread();
  /*if (firsttime) {
    if (rank == NULL) {
      rank = new int[TAU_MAX_THREADS];   
      firsttime = TauInitMpcThreads(rank);
    }
  }*/
  if (rank.size()<=tid||rank[tid] == -1) {
    while(rank.size()<=tid){
        rank.push_back(-1);
    }
    PMPI_Comm_rank(MPI_COMM_WORLD, &rank[tid]);
  }
  retval = rank[tid];
  RtsLayer::UnLockDB();

  return retval;
}
#else /* !TAU_MPC */

extern "C" int TauGetMpiRank(void)
{
#ifdef TAU_MPI
  int rank;
  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
#else
  return 0;
#endif /* TAU_MPI */
}
#endif /* TAU_MPC */
