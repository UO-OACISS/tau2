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

#include <mpi.h>
#include "mrnet/MRNet.h"

using namespace MRN;
using namespace std;

// Using a global for now. Could make it object-based like Aroon's old
//   codes later.
Network *net;
Stream *ctrl_stream;
// Determine whether to extend protocol to receive results from the FE.
bool broadcastResults;

extern "C" void Tau_mon_connect() {
  
  int rank;
  int size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  char myHostName[64];

  gethostname(myHostName, sizeof(myHostName));

  int targetRank;
  int beRank, mrnetPort, mrnetRank;
  char mrnetHostName[64];

  int in_beRank, in_mrnetPort, in_mrnetRank;
  char in_mrnetHostName[64];

  // Rank 0 reads in all connection information and sends appropriate
  //   chunks to the other ranks.
  if (rank == 0) {
    fprintf(stderr, "Connecting to ToM\n");

    FILE *connections = fopen("attachBE_connections","r");
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
	MPI_Send(&in_beRank, 1, MPI_INT, targetRank, 0, MPI_COMM_WORLD);
	MPI_Send(&in_mrnetPort, 1, MPI_INT, targetRank, 0, MPI_COMM_WORLD);
	MPI_Send(&in_mrnetRank, 1, MPI_INT, targetRank, 0, MPI_COMM_WORLD);
	MPI_Send(in_mrnetHostName, 64, MPI_CHAR, targetRank, 0, 
		 MPI_COMM_WORLD);
      }
    }
    fclose(connections);
  } else {
    MPI_Status status;
    MPI_Recv(&beRank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&mrnetPort, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&mrnetRank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(mrnetHostName, 64, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
  }

  int mrnet_argc = 6;
  char *mrnet_argv[6];

  char beRankString[10];
  char mrnetPortString[10];
  char mrnetRankString[10];

  sprintf(beRankString, "%d", beRank);
  sprintf(mrnetPortString, "%d", mrnetPort);
  sprintf(mrnetRankString, "%d", mrnetRank);

  mrnet_argv[0] = ""; // dummy process name string
  mrnet_argv[1] = mrnetHostName;
  mrnet_argv[2] = mrnetPortString;
  mrnet_argv[3] = mrnetRankString;
  mrnet_argv[4] = myHostName;
  mrnet_argv[5] = beRankString;

  net = Network::CreateNetworkBE(mrnet_argc, mrnet_argv);

  int tag;
  PacketPtr p;
  int data;

  // Do not proceed until control stream is established with front-end
  net->recv(&tag, p, &ctrl_stream);
  p->unpack("%d", &data);
  if (data == 1) {
    broadcastResults = true;
  } else if (data == 0) {
    broadcastResults = false;
  } else {
    fprintf(stderr,"Warning: Invalid initial control signal %d.\n",
	    data);
  }

  printf("[%d] Got ToM control stream. Proceeding with application code\n",
	 rank);
}

// more like a "last call" for protocol action than an
//   actual disconnect call. This call should not exit unless
//   the front-end says so.
extern "C" void Tau_mon_disconnect() {
  fprintf(stderr, "Disconnecting from ToM\n");

  // Tell front-end to tear down network and exit
  STREAM_FLUSHSEND(ctrl_stream, TOM_CONTROL, "%d", TOM_EXIT);
}

extern "C" void protocolLoop(int *globalToLocal, int numGlobal) {

  // receive from the network so that ToM will always know which stream to
  //   respond to.
  int protocolTag;
  PacketPtr p;
  Stream *stream;

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
    net->recv(&protocolTag, p, &stream);
    
    switch (protocolTag) {
    case PROT_BASESTATS: {
      printf("BE: Instructed by FE to report events and counters\n");
      // no need to unpack the data. Just send a response to the front-end
      STREAM_FLUSHSEND(stream, protocolTag, "%d %d", numGlobal, numCounters);

      // Remember, ToM always responds to the incoming stream.
      for (int evt=0; evt<numGlobal; evt++) {
	// printf("handling global event %d\n", evt);
	if (globalToLocal[evt] != -1) {
	  FunctionInfo *fi = TheFunctionDB()[globalToLocal[evt]];
	  for (int ctr=0; ctr<numCounters; ctr++) {
	    double sum = 0.0;
	    double sumofsqr = 0.0;
	    double min, max;
	    
	    net->recv(&protocolTag, p, &stream);
	    // accumulate for threads but contribute as a node
	    for (int tid=0; tid<numThreads; tid++) {
	      // going to work only with exclusive values for now.
	      double val = fi->getDumpExclusiveValues(tid)[ctr];
	      // printf("[%d,%d,%d] read exclusive value %f\n", 
	      //        evt, tid, ctr, val);
	      sum += val;
	      sumofsqr += val*val;
	      if (tid == 0) {
		min = val;
		max = val;
	      } else {
		if (min > val) {
		  min = val;
		}
		if (max < val) {
		max = val;
		}
	      }
	    }
	    // printf("[%d,%d] Sending %f %f %f %f %d %d\n", evt, ctr,
	    //        sum, sumofsqr, min, max, numThreads, numThreads);
	    // %f %f %f %f %d %d = sum sumofsqr min max contrib present
	    STREAM_FLUSHSEND(stream, protocolTag, "%lf %lf %lf %lf %d %d",
			     sum, sumofsqr, min, max, numThreads, numThreads);
	  }
	} else { /* globalToLocal[evt] != -1 */
	  // send a null contribution for each counter associated with
	  //   the function for this node
	  for (int ctr=0; ctr<numCounters; ctr++) {
	    net->recv(&protocolTag, p, &stream);
	    STREAM_FLUSHSEND(stream, protocolTag, "%lf %lf %lf %lf %d %d",
			     0.0, 0.0, 0.0, 0.0, 0, numThreads);
	  }
	} /* globalToLocal[evt] != -1 */
      }

      // Get results of the protocol from the front-end.
      if (broadcastResults) {
	int numMeans, numStdDevs, numMins, numMaxes;
	net->recv(&protocolTag, p, &stream);
	p->unpack("%alf %alf",
		  &means, &numMeans, &std_devs, &numStdDevs,
		  &mins, &numMins, &maxes, &numMaxes);
	/*
	printf("BE: Received %d values from FE\n", numMeans);
	for (int val=0; val<numMeans; val++) {
	  printf("BE: [%d] Mean:%f StdDev:%f\n", val, 
		 means[val], std_devs[val]);
	}
	*/
      }
      break;
    }
    case PROT_HIST: {
      int numBins;
      int numKeep;
      int numItems;
      int *keepItem;

      // in the case of the Histogramming Protocol, a percentage value
      //   is sent by the front-end to determine the threshold for
      //   exclusive execution time duration. Any event not satisfying
      //   the threshold will be filtered off.
      p->unpack("%ad %d %d", &keepItem, &numItems, &numKeep, &numBins);

      /* DEBUG
      printf("BE: Received filtered events from FE %d %d %d\n",
	     numItems, numKeep, numBins);
      for (int i=0; i<numItems; i++) {
	if (keepItem[i] == 1) {
	  printf("Keeping item %d\n", i);
	}
      }
      */

      // We send a set of bins for each counter + event combo that has
      //    not been filtered out.
      int *histBins;
      histBins = new int[numBins*numKeep];
      for (int i=0; i<numBins*numKeep; i++) {
	histBins[i] = 0;
      }


      int keepIdx = 0;
      for (int ctr=0; ctr<numCounters; ctr++) {
	for (int evt=0; evt<numGlobal; evt++) {
	  int aIdx = evt*numCounters+ctr;
	  if (keepItem[aIdx] == 1) {
	    if (globalToLocal[evt] != -1) {
	      FunctionInfo *fi = TheFunctionDB()[globalToLocal[evt]];
	      double range = maxes[aIdx] - mins[aIdx];
	      double interval = range/numBins;
	      // accumulate for threads but contribute as a node
	      for (int tid=0; tid<numThreads; tid++) {
		// going to work only with exclusive values for now.
		double val = fi->getDumpExclusiveValues(tid)[ctr];
		int histIdx = (int)floor((val-mins[aIdx])/interval);
		// hackish way of dealing with rounding problems.
		/*
		printf("BE: %f %f %f %f %d\n", 
		       interval, range, mins[aIdx], val, 
		       histIdx);
		*/
		if (histIdx < 0) {
		  histIdx = 0;
		}
		if (histIdx >= numBins) {
		  histIdx = numBins-1;
		}
		histBins[keepIdx*numBins+histIdx]++;
	      }
	    }
	    keepIdx++;
	  } 
	}
      }

      STREAM_FLUSHSEND(stream, protocolTag, "%ad", histBins, numKeep*numBins);
      break;
    }
    case PROT_CLASSIFIER: {
      break;
    }
    case PROT_CLUST_KMEANS: {
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
  printf("Protocol handled. Proceeding with application computation\n");
}

extern "C" void Tau_mon_onlineDump() {
  printf("Tau Mon data ready for dump.\n");

  // Need to get data loaded and computed from the stacks
  int numThreads = RtsLayer::getNumThreads();
  for (int tid = 0; tid<numThreads; tid++) {
    TauProfiler_updateIntermediateStatistics(tid);
  }

  // Unify events
  FunctionEventLister *functionEventLister = new FunctionEventLister();
  Tau_unify_object_t *functionUnifier = 
    Tau_unify_unifyEvents(functionEventLister);
  AtomicEventLister *atomicEventLister = new AtomicEventLister();
  Tau_unify_object_t *atomicUnifier = Tau_unify_unifyEvents(atomicEventLister);

  // process events in global order, pretty much the same way Collate handles
  //   the unified data.

  // the global number of events
  int numGlobal = functionUnifier->globalNumItems;
  int numLocal = functionUnifier->localNumItems;
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
    globalToLocal[functionUnifier->mapping[i]] = functionUnifier->sortMap[i];
  }

  /* DEBUG  
  for (int i=0; i<numGlobal; i++) {
    if (globalToLocal[i] == -1) {
      printf("No local event for global id %d\n", i);
    } else {
      FunctionInfo *fi = TheFunctionDB()[globalToLocal[i]];
      printf("Found idx %d: [%s] ExcT:%f\n", i, fi->GetName(),
	     fi->getDumpExclusiveValues(0)[0]);
    }
  }
  */

  // Tell the MRNet front-end that the data is ready and wait for
  //   protocol instructions.
  STREAM_FLUSHSEND(ctrl_stream, TOM_CONTROL, "%d", PROT_DATA_READY);
  protocolLoop(globalToLocal, numGlobal);
}

#endif /* TAU_EXP_UNIFY */
#endif /* TAU_MONITORING */
