// define these protocols only if monitoring is enabled. Otherwise,
//   TauMon.h will define default empty functions and we do not want
//   them interfering with one another.
#ifdef TAU_MONITORING

#include "Profile/TauMon.h"
#include "Profile/TauMonMrnet.h"

// for now, TAU_MONITORING will need to rely on TAU_EXP_UNIFY. This will
//   not be the case once event unification is implemented on every
//   available monitoring transport.
#ifdef TAU_EXP_UNIFY

#include <TAU.h>
#include <tau_types.h>
#include <TauUtil.h>
#include <TauMetrics.h>
#include <TauUnify.h>

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <mpi.h>

#ifdef MRNET_LIGHTWEIGHT
extern "C" {
#endif /* MRNET_LIGHTWEIGHT */

#ifdef MRNET_LIGHTWEIGHT
#include "mrnet_lightweight/MRNet.h"
#else /* MRNET_LIGHTWEIGHT */
#include "mrnet/MRNet.h"
#endif /* MRNET_LIGHTWEIGHT */

#ifndef MRNET_LIGHTWEIGHT
using namespace MRN;
using namespace std;
#endif
double tomGetData(int tomType, FunctionInfo *fi, int counter, int tid);
double tomGetFunData(int tomType, FunctionInfo *fi, int tid);
double tomGetCtrData(int tomType, FunctionInfo *fi, int counter, int tid);
int calcHistBinIdx(int numBins, double val, double max, double min);
	           
double calcEuclideanDistance(double *vector, double *centroids,
			     int numEvents, int k);

// Back-end rank
int rank;

// Using a global for now. Could make it object-based like Aroon's old
//   codes later.
#ifdef MRNET_LIGHTWEIGHT
Network_t *net;
Stream_t *ctrl_stream = (Stream_t *)malloc(sizeof(Stream_t));
#else /* MRNET_LIGHTWEIGHT */
Network *net;
Stream *ctrl_stream;
#endif /* MRNET_LIGHTWEIGHT */

// Determine whether to extend protocol to receive results from the FE.
bool broadcastResults;

const char *profiledir;

// Unification structures
FunctionEventLister *mrnetFuncEventLister;
Tau_unify_object_t *mrnetFuncUnifier;
AtomicEventLister *mrnetAtomEventLister;
Tau_unify_object_t *mrnetAtomUnifier;

extern "C" void calculateStats(double *sum, double *sumofsqr, 
			       double *max, double *min,
			       double value);

extern "C" void Tau_mon_connect() {

  //  printf("Mon Connect called\n");
  
  int size;

  PMPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PMPI_Comm_size(MPI_COMM_WORLD, &size);

  char myHostName[64];

  gethostname(myHostName, sizeof(myHostName));
  profiledir = TauEnv_get_profiledir();

  int targetRank;
  int beRank, mrnetPort, mrnetRank;
  char mrnetHostName[64];

  int in_beRank, in_mrnetPort, in_mrnetRank;
  char in_mrnetHostName[64];

  // Rank 0 reads in all connection information and sends appropriate
  //   chunks to the other ranks.
  if (rank == 0) {
    TAU_VERBOSE("Connecting to ToM\n");

    // Do not proceed until front-end has written the atomic probe file.
    char atomicFileName[512];
    sprintf(atomicFileName,"%s/ToM_FE_Atomic",profiledir);
    FILE *atomicFile;
    while ((atomicFile = fopen(atomicFileName,"r")) == NULL) {
      sleep(1);
    }
    fclose(atomicFile);

    char connectionName[512];
    sprintf(connectionName,"%s/attachBE_connections",profiledir);
    FILE *connections = fopen(connectionName,"r");
    // assume there are exactly size entries in the connection file.
    for (int i=0; i<size; i++) {
      fscanf(connections, "%d %d %d %d %s\n",
	     &targetRank, &in_beRank, &in_mrnetPort, &in_mrnetRank,
	     in_mrnetHostName);
      if (targetRank == 0) {
	beRank = in_beRank;
	mrnetPort = in_mrnetPort;
	mrnetRank = in_mrnetRank;
	strncpy(mrnetHostName, in_mrnetHostName, 64);
      } else {
	PMPI_Send(&in_beRank, 1, MPI_INT, targetRank, 0, MPI_COMM_WORLD);
	PMPI_Send(&in_mrnetPort, 1, MPI_INT, targetRank, 0, MPI_COMM_WORLD);
	PMPI_Send(&in_mrnetRank, 1, MPI_INT, targetRank, 0, MPI_COMM_WORLD);
	PMPI_Send(in_mrnetHostName, 64, MPI_CHAR, targetRank, 0, 
		 MPI_COMM_WORLD);
      }
    }
    fclose(connections);
  } else {
    MPI_Status status;
    PMPI_Recv(&beRank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    PMPI_Recv(&mrnetPort, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    PMPI_Recv(&mrnetRank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    PMPI_Recv(mrnetHostName, 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
  }

  int mrnet_argc = 6;
  char *mrnet_argv[6];

  char beRankString[10];
  char mrnetPortString[10];
  char mrnetRankString[10];

  sprintf(beRankString, "%d", beRank);
  sprintf(mrnetPortString, "%d", mrnetPort);
  sprintf(mrnetRankString, "%d", mrnetRank);

  mrnet_argv[0] = (char *)malloc(strlen("")*sizeof(char));
  mrnet_argv[0] = strcpy(mrnet_argv[0],""); // dummy process name string
  mrnet_argv[1] = mrnetHostName;
  mrnet_argv[2] = mrnetPortString;
  mrnet_argv[3] = mrnetRankString;
  mrnet_argv[4] = myHostName;
  mrnet_argv[5] = beRankString;

#ifdef MRNET_LIGHTWEIGHT
  net = Network_CreateNetworkBE(mrnet_argc, mrnet_argv);
  assert(net);
#else /* MRNET_LIGHTWEIGHT */
  net = Network::CreateNetworkBE(mrnet_argc, mrnet_argv);
#endif /* MRNET_LIGHTWEIGHT */

  int tag;
  int data;

// Do not proceed until control stream is established with front-end
#ifdef MRNET_LIGHTWEIGHT
  Packet_t *p = (Packet_t *)malloc(sizeof(Packet_t));
  Network_recv(net, &tag, p, &ctrl_stream);
  Packet_unpack(p, "%d", &data);
#else /* MRNET_LIGHTWEIGHT */
  PacketPtr p;
  net->recv(&tag, p, &ctrl_stream);
  p->unpack("%d", &data);
#endif /* MRNET_LIGHTWEIGHT */

  if (data == 1) {
    broadcastResults = true;
  } else if (data == 0) {
    broadcastResults = false;
  } else {
    fprintf(stderr,"Warning: Invalid initial control signal %d.\n",
	    data);
  }
}

// more like a "last call" for protocol action than an
//   actual disconnect call. This call should not exit unless
//   the front-end says so.
extern "C" void Tau_mon_disconnect() {
  if (rank == 0) {
    TAU_VERBOSE("Disconnecting from ToM\n");
  }
  // Tell front-end to tear down network and exit
  STREAM_FLUSHSEND_BE(ctrl_stream, TOM_CONTROL, "%d", TOM_EXIT);
}

extern "C" void protocolLoop(int *globalToLocal, int numGlobal) {

  // receive from the network so that ToM will always know which stream to
  //   respond to.
  int protocolTag;
#ifdef MRNET_LIGHTWEIGHT
  Packet_t *p = (Packet_t *)malloc(sizeof(Packet_t));
  Stream_t *stream = (Stream_t *)malloc(sizeof(Stream_t));
#else /* MRNET_LIGHTWEIGHT */
  PacketPtr p;
  Stream *stream;
#endif /* MRNET_LIGHTWEIGHT */

  bool processProtocol = true;

  // data from Basestats to be kept for later procotols. *CWL* find a
  //   modular way for this information to be exchanged between protocols.
  double *means;
  double *std_devs;  
  double *mins;
  double *maxes;

  int numThreads = RtsLayer::getNumThreads();
  int numCounters = Tau_Global_numCounters;

  while (processProtocol) {
#ifdef MRNET_LIGHTWEIGHT
    Network_recv(net, &protocolTag, p, &stream);
#else /* MRNET_LIGHTWEIGHT */
    net->recv(&protocolTag, p, &stream);
#endif /* MRNET_LIGHTWEIGHT */

    switch (protocolTag) {
    case PROT_UNIFY: {
      // Rank 0 additionally responds to the front-end with all global
      //   function name strings.
      if (rank == 0) {
	int tag;
#ifdef MRNET_LIGHTWEIGHT
	Packet_t *p = (Packet_t *)malloc(sizeof(Packet_t));    
#else /* MRNET_LIGHTWEIGHT */
	PacketPtr p;
#endif /* MRNET_LIGHTWEIGHT */
	printf("Num Global = %d\n", numGlobal);
#ifdef MRNET_LIGHTWEIGHT
	Network_recv(net, &tag, p, &stream);
#else /* MRNET_LIGHTWEIGHT */
        net->recv(&tag, p, &stream);
#endif /* MRNET_LIGHTWEIGHT */
	STREAM_FLUSHSEND_BE(stream, PROT_UNIFY, "%as",
			 mrnetFuncUnifier->globalStrings, numGlobal);
      }
      break;
    }
    case PROT_BASESTATS: {
      // *DEBUG* 
      if (rank == 0) {
	printf("BE: Instructed by FE to report events and counters.\n");
      }
      // First message is a request for names
      // Invoke Unification
      FunctionEventLister *functionEventLister = new FunctionEventLister();
      Tau_unify_object_t *functionUnifier = 
	Tau_unify_unifyEvents(functionEventLister);
      // Send Names of events.
      char **tomNames;
      if (rank == 0) {
	// send the array of event name strings.
	tomNames = (char **)malloc((numCounters+numGlobal)*sizeof(char*));
	for (int m=0; m<numCounters; m++) {
	  tomNames[m] =
	    (char *)malloc((strlen(TauMetrics_getMetricName(m))+1)*
			   sizeof(char));
	  strcpy(tomNames[m],TauMetrics_getMetricName(m));
	}
	for (int f=0; f<numGlobal; f++) {
	  tomNames[f+numCounters] = 
	    (char *)malloc((strlen(functionUnifier->globalStrings[f])+1)*
			   sizeof(char));
	  strcpy(tomNames[f+numCounters],functionUnifier->globalStrings[f]);
	}
	STREAM_FLUSHSEND_BE(stream, PROT_BASESTATS, "%d %d %as",
			    rank, numCounters, tomNames, 
			    numCounters+numGlobal);
      } else {
	// send a single null string over.
	tomNames = (char **)malloc(sizeof(char*));
	tomNames[0] = (char *)malloc(sizeof(char));
	strcpy(tomNames[0],"");
	STREAM_FLUSHSEND_BE(stream, PROT_BASESTATS, "%d %d %as",
			    rank, 0, tomNames, 1);
      }

      // Then receive request for stats. No need to unpack.
      int tag;
#ifdef MRNET_LIGHTWEIGHT
      Network_recv(net, &tag, p, &stream);
#else /* MRNET_LIGHTWEIGHT */
      net->recv(&tag, p, &stream);
#endif /* MRNET_LIGHTWEIGHT */
      assert(tag == PROT_BASESTATS);

      int numItems = numCounters*TOM_NUM_CTR_VAL+TOM_NUM_FUN_VAL;
      int dataLength = numGlobal*numItems;

      double *out_sums;
      double *out_sumsofsqr;
      double *out_mins;
      double *out_maxes;
      out_sums = (double *)TAU_UTIL_MALLOC(dataLength*sizeof(double));
      out_sumsofsqr = (double *)TAU_UTIL_MALLOC(dataLength*sizeof(double));
      out_mins = (double *)TAU_UTIL_MALLOC(dataLength*sizeof(double));
      out_maxes = (double *)TAU_UTIL_MALLOC(dataLength*sizeof(double));
      for (int i=0; i<dataLength; i++) {
	out_sums[i] = 0.0;
	out_sumsofsqr[i] = 0.0;
	out_mins[i] = 0.0;
	out_maxes[i] = 0.0;
      }

      // Construct the data arrays to be sent.
      for (int evt=0; evt<numGlobal; evt++) {
	if (globalToLocal[evt] != -1) {
	  FunctionInfo *fi = TheFunctionDB()[globalToLocal[evt]];
	  // CALL and SUBR values are event-specific, not counter-related.
	  int offset = evt*numItems;
	  for (int type=0; type<TOM_NUM_FUN_VAL; type++) {
	    int finalIdx = offset+type;
	    for (int tid=0; tid<numThreads; tid++) {
	      double val = tomGetFunData(type, fi, tid);
	      if (tid == 0) {
		out_mins[finalIdx] = val;
		out_maxes[finalIdx] = val;
	      }
	      // the counter value is ignored in this case.
	      calculateStats(&out_sums[finalIdx], &out_sumsofsqr[finalIdx],
			     &out_mins[finalIdx], &out_maxes[finalIdx],
			     val);
	    }
	  }
	  // Each counter has EXCL and INCL values
	  for (int counter=0; counter<numCounters; counter++) {
	    // accumulate for threads but contribute as a node
	    // Write thread-accumulated values into the appropriate arrays
	    int offset = 
	      evt*numItems + TOM_NUM_FUN_VAL + counter*TOM_NUM_CTR_VAL;
	    for (int type=0; type<TOM_NUM_CTR_VAL; type++) {
	      int finalIdx = offset+type;
	      for (int tid=0; tid<numThreads; tid++) {
		double val = tomGetCtrData(type, fi, counter, tid);
		if (tid == 0) {
		  out_mins[finalIdx] = val;
		  out_maxes[finalIdx] = val;
		}
		calculateStats(&out_sums[finalIdx], &out_sumsofsqr[finalIdx],
			       &out_mins[finalIdx], &out_maxes[finalIdx],
			       val);
	      }
	    }
	  }
	} /* globalToLocal[evt] != -1 */
      }

      STREAM_FLUSHSEND_BE(stream, protocolTag, 
			  "%d %d %alf %alf %alf %alf %d",
			  numGlobal, numCounters,
			  out_sums, dataLength,
			  out_sumsofsqr, dataLength,
			  out_mins, dataLength,
			  out_maxes, dataLength,
			  numThreads);

      // Get results of the protocol from the front-end.
      if (broadcastResults) {
	int numMeans, numStdDevs, numMins, numMaxes;
#ifdef MRNET_LIGHTWEIGHT
	Network_recv(net, &protocolTag, p, &stream);
	Packet_unpack(p, "%alf %alf %alf %alf",
		      &means, &numMeans, &std_devs, &numStdDevs,
		      &mins, &numMins, &maxes, &numMaxes);
#else /* MRNET_LIGHTWEIGHT */
        net->recv(&protocolTag, p, &stream);
        p->unpack("%alf %alf %alf %alf",
                  &means, &numMeans, &std_devs, &numStdDevs,
                  &mins, &numMins, &maxes, &numMaxes);
#endif /* MRNET_LIGHTWEIGHT */
	//	printf("Received %d broadcast results\n",numMeans);
      }
      break;
    }
    case PROT_HIST: {
      int numBins;
      // in the case of the Histogramming Protocol, a percentage value
      //   is sent by the front-end to determine the threshold for
      //   exclusive execution time duration. Any event not satisfying
      //   the threshold will be filtered off.
#ifdef MRNET_LIGHTWEIGHT
      Packet_unpack(p, "%d", &numBins);
#else /* MRNET_LIGHTWEIGHT */
      p->unpack("%d", &numBins);
#endif /* MRNET_LIGHTWEIGHT */

      int numHistogramsPerEvent = 
	numCounters*TOM_NUM_CTR_VAL+TOM_NUM_FUN_VAL;
      int numItems = numGlobal*numHistogramsPerEvent;
      int dataLength = numBins*numItems;

      int *histBins;
      histBins = new int[dataLength];
      for (int i=0; i<dataLength; i++) {
	histBins[i] = 0;
      }

      for (int evt=0; evt<numGlobal; evt++) {
	FunctionInfo *fi = TheFunctionDB()[globalToLocal[evt]];
	if (globalToLocal[evt] != -1) {
	  // CALL and SUBR applies only to events
	  int dataIdx = evt*numHistogramsPerEvent;
	  for (int type=0; type<TOM_NUM_FUN_VAL; type++) {
	    int finalDataIdx = dataIdx + type;
	    for (int tid=0; tid<numThreads; tid++) {
	      double val = tomGetFunData(type, fi, tid);
	      int binIdx = 
		calcHistBinIdx(numBins, val, 
			       maxes[finalDataIdx], mins[finalDataIdx]);
	      histBins[(finalDataIdx*numBins)+binIdx]++;
	    }
	  }
	  // EXCL and INCL values for each counter
	  for (int counter=0; counter<numCounters; counter++) {
	    int dataIdx = 
	      evt*numHistogramsPerEvent + TOM_NUM_FUN_VAL +
	      counter*TOM_NUM_CTR_VAL;
	    int histIdx = evt*numHistogramsPerEvent + counter*2;
	    for (int type=0; type<TOM_NUM_CTR_VAL; type++) {
	      int finalDataIdx = dataIdx + type;
	      for (int tid=0; tid<numThreads; tid++) {
		double val = tomGetCtrData(type, fi, counter, tid);
		int binIdx = 
		  calcHistBinIdx(numBins, val, 
				 maxes[finalDataIdx], mins[finalDataIdx]);
		histBins[(finalDataIdx*numBins)+binIdx]++;
	      }
	    }
	  }
	}
      }
      //      printf("Sending Histogram results\n");
      STREAM_FLUSHSEND_BE(stream, protocolTag, "%ad", histBins, 
			  dataLength);
      break;
    }
    case PROT_CLASSIFIER: {
      break;
    }
    case PROT_CLUST_KMEANS: {
      int numK;
      // react to frontend
#ifdef MRNET_LIGHTWEIGHT
      Packet_unpack(p, "%d", &numK);
#else /* MRNET_LIGHTWEIGHT */
      p->unpack("%d", &numK);
#endif /* MRNET_LIGHTWEIGHT */

      // send acknowledgement via received stream (control stream)
      STREAM_FLUSHSEND_BE(stream, protocolTag, "%d", numK);

      // For now, work with only Exclusive TIME WITHOUT threads.
      // int numItemsPerEvent = (numCounters*TOM_NUM_CTR_VAL)+TOM_NUM_FUN_VAL;
      int numItemsPerEvent = 1;
      int numItemsPerK = numGlobal*numItemsPerEvent;
      int dataLength = numK*numItemsPerK;

      double *globalClusterCentroids;

      double *myVectors; // of size numItemsPerK
      int myK;
      int newK;
      double *changeVectors; // of size numK*numItemsPerK
      int *numMembers; // of size numK*numItemsPerEvent

      // Collect performance information on each thread.
      //    When pertinent, apply normalization factors to the vectors.
      // Note that threads have to be dealt with separately here. 
      myVectors = new double[numItemsPerK];
      for (int i=0; i<numItemsPerK; i++) {
	myVectors[i] = 0.0;
      }
      for (int evt=0; evt<numGlobal; evt++) {
	if (globalToLocal[evt] != -1) {
	  FunctionInfo *fi = TheFunctionDB()[globalToLocal[evt]];
	  myVectors[evt] = tomGetCtrData(TOM_CTR_EXCL, fi, 0, 0);
	}
      }

      // receive initial k choices.
      int numInItems;
      int *choices;
#ifdef MRNET_LIGHTWEIGHT
      Network_recv(net, &protocolTag, p, &stream);
      Packet_unpack(p, "%ad", &choices, &numInItems);
#else /* MRNET_LIGHTWEIGHT */
      net->recv(&protocolTag, p, &stream);
      p->unpack("%ad", &choices, &numInItems);
#endif /* MRNET_LIGHTWEIGHT */

      // Decide if I am one of the selected k participants.
      myK = -1;  // out-of-band value
      newK = -1;
      for (int k=0; k<numK; k++) {
	if (choices[k] == rank) {
	  myK = k;
	}
      }

      // Update change vectors. This special initial case handles myK
      //   as out-of-band values. Note that numMembers does not matter
      //   in this case, so set all values to 0.
      changeVectors = new double[numK*numItemsPerK];
      numMembers = new int[numK*numItemsPerEvent];
      for (int k=0; k<numK; k++) {
	numMembers[k] = 0;
	for (int evt=0; evt<numGlobal; evt++) {
	  changeVectors[k*numGlobal+evt] =
	    (myK == k) ? myVectors[evt] : 0.0;
	}
      }
      
      // Report change vectors back to root.
      STREAM_FLUSHSEND_BE(stream, protocolTag, "%alf %ad", 
			  changeVectors, numK*numItemsPerK,
			  numMembers, numK*numItemsPerEvent);

      // Loop until done.
      bool initial = true; // for handling special conditions
      bool stop = false;
      int stopTag;
      do {
#ifdef MRNET_LIGHTWEIGHT
	Network_recv(net, &protocolTag, p, &stream);
	Packet_unpack(p, "%alf", &globalClusterCentroids, &numInItems);
#else /* MRNET_LIGHTWEIGHT */
	net->recv(&protocolTag, p, &stream);
	p->unpack("%alf", &globalClusterCentroids, &numInItems);
#endif /* MRNET_LIGHTWEIGHT */
	
	// sanity check
	assert(numInItems == dataLength);

	int testRank = 2;
	if (rank == testRank) {
	  printf("[%d] Centroids = ", rank);
	  for (int i=0; i<numInItems; i++) {
	    printf("%.16G ",globalClusterCentroids[i]);
	  }
	  printf("\n");
	  printf("[%d] My vector = ", rank);
	  for (int i=0; i<numItemsPerK; i++) {
	    printf("%.16G ",myVectors[i]);
	  }
	  printf("\n");
	}

	// decide which k the node's vector belongs to.
	double minDist = 0.0;
	for (int k=0; k<numK; k++) {
	  double distance =
	    calcEuclideanDistance(myVectors, globalClusterCentroids,
				  numGlobal, k);
	  if (k == 0) {
	    minDist = distance;
	    newK = k;
	  } else {
	    if (distance < minDist) {
	      minDist = distance;
	      newK = k;
	    }
	  }
	  if (rank == testRank) {
	    printf("Curr %d: %.16G newK=%d\n",k, distance, newK);
	  }
	}
	
	// Update change vectors.
	for (int k=0; k<numK; k++) {
	  numMembers[k] = 0;
	  for (int evt=0; evt<numGlobal; evt++) {
	    changeVectors[k*numGlobal+evt] = 0.0;
	  }
	}
	// did anything change?
	if (!initial) {
	  if (myK != newK) {
	    numMembers[myK] = -1;
	    numMembers[newK] = 1;
	    for (int evt=0; evt<numGlobal; evt++) {
	      changeVectors[myK*numGlobal+evt] = -myVectors[evt];
	      changeVectors[newK*numGlobal+evt] = myVectors[evt];
	    }
	  }
	} else {
	  numMembers[newK] = 1;
	  for (int evt=0; evt<numGlobal; evt++) {
	    changeVectors[newK*numGlobal+evt] = myVectors[evt];
	  }
	}

	if (rank == testRank) {
	  printf("[%d] My Change contribution: ", rank);
	  for (int i=0; i<numK*numItemsPerK; i++) {
	    printf("%.16G ", changeVectors[i]);
	  }
	  printf("\n");
	  for (int i=0; i<numK*numItemsPerEvent; i++) {
	    printf("%d ", numMembers[i]);
	  }
	  printf("\n");
	}

	// modify myK
	myK = newK;
	newK = -1; // extra insurance against error
	if (initial) {
	  initial = false;
	}

	// - send the change-vector
	STREAM_FLUSHSEND_BE(stream, protocolTag, "%alf %ad", 
			    changeVectors, numK*numItemsPerK,
			    numMembers, numK*numItemsPerEvent);

	//   - receive confirmation we can stop.
#ifdef MRNET_LIGHTWEIGHT
	Network_recv(net, &protocolTag, p, &stream);
	Packet_unpack(p, "%d", &stopTag);
#else /* MRNET_LIGHTWEIGHT */
	net->recv(&protocolTag, p, &stream);
	p->unpack("%d", &stopTag);
#endif /* MRNET_LIGHTWEIGHT */
	//  Acknowledge convergence flag
	STREAM_FLUSHSEND_BE(stream, protocolTag, "%d", stopTag);

	if (stopTag != 0) {
	  stop = true;
	}
      } while (!stop);
      break;
    }
    case PROT_EXIT: {
      // Front-end is done with this round of the processing protocol.
      //   Backends are now allowed to proceed with computation.
      processProtocol = false;
      break;
    }
    default: {
      printf("Warning: Unknown protocol tag [%d]\n", protocolTag);
    }
    }
  } /* while (processProtocol) */
  if (rank == 0) {
    TAU_VERBOSE("BE: Protocol handled. Application computation follows.\n");
  }
}

extern "C" void Tau_mon_onlineDump() {
  // *DEBUG* printf("Tau Mon data ready for dump.\n");

  // Need to get data loaded and computed from the stacks
  int numThreads = RtsLayer::getNumThreads();
  for (int tid = 0; tid<numThreads; tid++) {
    TauProfiler_updateIntermediateStatistics(tid);
  }

  // Unify events
  mrnetFuncEventLister = new FunctionEventLister();
  mrnetFuncUnifier = Tau_unify_unifyEvents(mrnetFuncEventLister);
  mrnetAtomEventLister = new AtomicEventLister();
  mrnetAtomUnifier = Tau_unify_unifyEvents(mrnetAtomEventLister);

  // process events in global order, pretty much the same way Collate handles
  //   the unified data.

  // the global number of events
  int numGlobal = mrnetFuncUnifier->globalNumItems;
  int numLocal = mrnetFuncUnifier->localNumItems;
  assert(numLocal <= numGlobal);
  int *globalToLocal = 
    (int*)TAU_UTIL_MALLOC(numGlobal*sizeof(int));
  // initialize all global entries to -1
  //   where -1 indicates that the event did not occur for this rank
  for (int i=0; i<numGlobal; i++) { 
    globalToLocal[i] = -1; 
  }
  for (int i=0; i<numLocal; i++) {
    // set reverse unsorted mapping
    globalToLocal[mrnetFuncUnifier->mapping[i]] = mrnetFuncUnifier->sortMap[i];
  }
  // Tell the MRNet front-end that the data is ready and wait for
  //   protocol instructions.
  STREAM_FLUSHSEND_BE(ctrl_stream, TOM_CONTROL, "%d", PROT_DATA_READY);

  // Start the protocol loop.
  protocolLoop(globalToLocal, numGlobal);
}

void calculateStats(double *sum, double *sumofsqr, 
		    double *max, double *min,
		    double val) {
  *sum += val;
  *sumofsqr += val*val;
  if (*min > val) {
    *min = val;
  }
  if (*max < val) {
    *max = val;
  }
}

double tomGetFunData(int tomType, FunctionInfo *fi, int tid) {
  double val = -1.0; // out-of-band value
  switch (tomType) {
  case TOM_FUN_CALL: {
    val = fi->GetCalls(tid)*1.0;
    break;
  }
  case TOM_FUN_SUBR: {
    val = fi->GetSubrs(tid)*1.0;
    break;
  }
  }
  return val;
}

double tomGetCtrData(int tomType, FunctionInfo *fi, int counter, int tid) {
  double val = -1.0; // out-of-band value
  switch (tomType) {
  case TOM_CTR_INCL: {
    val = fi->getDumpInclusiveValues(tid)[counter];
    break;
  }
  case TOM_CTR_EXCL: {
    val = fi->getDumpExclusiveValues(tid)[counter];
    break;
  }
  }
  return val;
}

double tomGetData(int tomType, FunctionInfo *fi, int counter, int tid) {
  double val = -1.0; // out-of-band value
  switch (tomType) {
  case TOM_VAL_INCL: {
    val = fi->getDumpInclusiveValues(tid)[counter];
    break;
  }
  case TOM_VAL_EXCL: {
    val = fi->getDumpExclusiveValues(tid)[counter];
    break;
  }
  case TOM_VAL_CALL: {
    val = fi->GetCalls(tid)*1.0;
    break;
  }
  case TOM_VAL_SUBR: {
    val = fi->GetSubrs(tid)*1.0;
    break;
  }
  }
  return val;
}

int calcHistBinIdx(int numBins, double val, double max, double min) {
  double range = max - min;
  double interval = range/numBins;
  int binIdx = 
    (int)floor((val-min)/interval);
  // hackish way of dealing with rounding problems.
  if (binIdx < 0) {
    binIdx = 0;
  }
  if (binIdx >= numBins) {
    binIdx = numBins-1;
  }
  return binIdx;
}

void updateChangeVector(double **changeVector, double *myVectors,
			int oldK, int newK,
			int tid, int ctr, int type,
			int numThreads, int numEvents, 
			int numCounters, int numK) {
  int numItemsPerEvent = numCounters*TOM_NUM_CTR_VAL+TOM_NUM_FUN_VAL;
}

double calcEuclideanDistance(double *vector, double *centroids,
			     int numEvents, int k) {
  double distance = 0.0;
  for (int evt=0; evt<numEvents; evt++) {
    distance += pow(vector[evt] - 
		    centroids[k*numEvents+evt], 2.0);
  }  
  return sqrt(distance);
}

#ifdef MRNET_LIGHTWEIGHT
} /* extern "C" */
#endif /* MRNET_LIGHTWEIGHT */

#endif /* TAU_EXP_UNIFY */
#endif /* TAU_MONITORING */
