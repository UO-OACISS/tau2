/****************************************************************************
 * Copyright © 2003-2009 Dorian C. Arnold, Philip C. Roth, Barton P. Miller *
 *                  Detailed MRNet usage rights in "LICENSE" file.          *
 ****************************************************************************/

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <sys/time.h>

#include <stdint.h>
#include <string.h>

#include "mrnet/MRNet.h"
#include "Profile/TauMonMrnet.h"

using namespace MRN;
using namespace std;

Network *net = NULL;
Communicator *comm_BC;

Stream *ctrl_stream;

int unifyFilterId;
int baseStatsFilterId;
int baseStatsNameFilterId;
int histogramFilterId;

bool broadcastResults;

// *CWL* General purpose filters?
int syncFilterId;

char *profiledir;

// Shared information between Base Stats and Histogram protocols
// *CWL* - find a better way of implementing this, probably through
//    object-based interfaces. 
int numEvents, numCounters;
double *means;
char **eventNames;
char **tomNames;
int numMetrics;
int globalNumThreads;
char **counterNames;
// map from BE ranks to MPI ranks
int *rankMap;

int invocationIndex;

// Timer variables
double time_aggregate;
double time_hist;

double ToM_getTimeOfDay();

void controlLoop();

// early idea for eventual modularization
void registerProtocols();
void registerUnify();
void registerBaseStats();
void registerHistogram();

void monitoringProtocol();
void builtInProtocols();
void protocolUnify();
void protocolBaseStats();
void protocolHistogram();

void write_be_connections(vector<NetworkTopology::Node *>& leaves, 
			  int num_net_nodes,
			  int num_be)
{
   FILE *f;

   char connfile[512];
   sprintf(connfile,"%s/attachBE_connections",profiledir);
   if ((f = fopen(connfile, (const char *)"w+")) == NULL)
   {
      perror("fopen");
      exit(-1);
   }

   // block-assignment of back-ends to leaf nodes
   rankMap = new int[num_be];
   int num_leaves = leaves.size();
   int be_per_leaf = (int)(ceil((double)num_be/(double)num_leaves));
   int remaining_be = num_be;
   for (int i=0; i<num_leaves; i++) {
     int be_to_leaf = (remaining_be >= be_per_leaf)?be_per_leaf:remaining_be;
     for (int be_offset=0; be_offset<be_to_leaf; be_offset++) {
       int be_real_rank = num_be - remaining_be + be_offset;
       int be_mrnet_rank = be_real_rank + num_net_nodes;
       rankMap[be_real_rank] = be_mrnet_rank;
       fprintf(f, "%d %d %d %d %s\n", 
	       be_real_rank,
	       be_mrnet_rank,
	       leaves[i]->get_Port(), 
	       leaves[i]->get_Rank(),
	       leaves[i]->get_HostName().c_str());
     }
     remaining_be -= be_to_leaf;
   }
   fclose(f);
}

int main(int argc, char **argv)
{
    int num_mrnet_nodes = 0;

    if (argc != 3) {
      fprintf(stderr, "Usage: %s <topology_file> <num_backends>\n", argv[0]);
      exit(-1);
    }

    profiledir = getenv("PROFILEDIR");
    if (profiledir == NULL) {
      profiledir = (char *)malloc((strlen(".")+1)*sizeof(char));
      strcpy(profiledir,".");
    }

    char* topology_file = argv[1];
    int num_backends = atoi(argv[2]);

    // *CWL* can make this a switch next time.
    broadcastResults = true;

    printf("FE: Creating network\n");
    
    // If backend_exe (2nd arg) and backend_args (3rd arg) are both NULL,
    // then all nodes specified in the topology are internal tree nodes.
    net = Network::CreateNetworkFE(topology_file, NULL, NULL);

    printf("FE: Network created\n");

    // Load filter functions now (need more elegant way later)
    registerProtocols();
    syncFilterId = net->load_FilterFunc("ToM_Sync_Filter.so",
					"ToM_Sync_Filter");
  
    // Query net for topology object
    NetworkTopology *topology = net->get_NetworkTopology();
    num_mrnet_nodes = topology->get_NumNodes();
    vector< NetworkTopology::Node * > internal_leaves;
    topology->get_Leaves(internal_leaves);

    // Write connection information to temporary file
    write_be_connections(internal_leaves, num_mrnet_nodes, num_backends);

    printf("FE: MRNet network successfully created.\n");
    printf("FE: Waiting for %u backends to connect.\n", num_backends );
    fflush(stdout);

    // Write an atomic probe file for Backends to wait on.
    FILE *atomicFile;
    char atomicFilename[512];
    sprintf(atomicFilename,"%s/ToM_FE_Atomic",profiledir);
    if ((atomicFile = fopen(atomicFilename,"w")) == NULL) {
      perror("Failed to create ToM_FE_Atomic\n");
      exit(-1);
    } else {
      fclose(atomicFile);
    }

    set<NetworkTopology::Node *> be_nodes;
    do {
        sleep(1);
	topology->get_BackEndNodes(be_nodes);
    } while (be_nodes.size() < num_backends);
    printf("FE: All application backends connected!\n");

    // Specialized stream construction
    // Broadcast stream
    comm_BC = net->get_BroadcastCommunicator();
    ctrl_stream = net->new_Stream(comm_BC, syncFilterId);

    // should backends go away?
    net->set_TerminateBackEndsOnShutdown(true);

    printf("FE: Inform back-ends of the control streams to use\n");
    STREAM_FLUSHSEND(ctrl_stream, TOM_CONTROL, "%d", broadcastResults?1:0);

    // control loop
    controlLoop();

    printf("FE: Done.\n");
    return 0;
}

void controlLoop() {

  bool processProtocol = true;

  invocationIndex = 0;
  while (processProtocol) {
    // Check only on the Control stream. No other packets should come
    //   up the tree outside of their individual protocol channels.
    int tag;
    PacketPtr p;
    int protocolTag;

    printf("FE: Waiting on next back-end initiated signal\n");
    ctrl_stream->recv(&tag, p);
    // process packet and decide which protocol pattern to invoke
    //   for the desired response.
    p->unpack("%d", &protocolTag);

    switch (protocolTag) {
    case TOM_EXIT: {
      // The Network destructor causes internal and leaf nodes to exit
      // Wait for all backends to go away before initiating the teardown
      //   of the network.
      NetworkTopology *topology = net->get_NetworkTopology();
      set<NetworkTopology::Node *> be_nodes;

      // This is hackish. Give the backends some time to complete
      //   (after MPI_Finalize) before attempting to tear down
      //   the MRNet network (which is supposed to terminate the
      //   backends as part of the process).
      sleep(10);
      printf("FE: Tearing down MRNet network.\n");
      delete net;

      // This is hackish. Give the comm nodes some time to properly
      //    shutdown before letting the front-end process die.
      sleep(10);
      printf("FE: Shutdown after net delete.\n");

      processProtocol = false;
      break;
    }
    case PROT_DATA_READY: {
      printf("FE: Data ready at application backends. Start protocols.\n");
      monitoringProtocol();
      invocationIndex++;
      break;
    }
    default: {
      printf("Warning: Unknown protocol tag [%d]\n", protocolTag);
    }
    }
    
  }
}

void registerProtocols() {
  // registerUnify();
  registerBaseStats();
  registerHistogram();
}

void registerUnify() {
  unifyFilterId = net->load_FilterFunc("ToM_Unification_Filter.so",
				       "ToM_Unification_Filter");
}

void registerBaseStats() {
  baseStatsFilterId = net->load_FilterFunc("ToM_Stats_Filter.so",
					   "ToM_Stats_Filter");
  baseStatsNameFilterId = net->load_FilterFunc("ToM_Name_Filter.so",
					       "ToM_Name_Filter");
}

void registerHistogram() {
  // load the Histogram filter here.
  histogramFilterId = net->load_FilterFunc("ToM_Histogram_Filter.so",
					   "ToM_Histogram_Filter");
}

void monitoringProtocol() {
  // basically activate all desired protocols on ready signal 
  // from application
  builtInProtocols();
  STREAM_FLUSHSEND(ctrl_stream, PROT_EXIT, "%d", PROT_EXIT);
}

void builtInProtocols() {
  //  protocolUnify();
  // **CWL** Consider a way to turn them on or off, even dynamically.
  protocolBaseStats();
  protocolHistogram();
}

void protocolUnify() {
  int tag;
  PacketPtr p;
  
  // ask application for name strings. These can be acquired from
  //   Rank 0 after MPI-based unification.
  int numRecvEvents;
  p->unpack("%as", &eventNames, &numRecvEvents);
  printf("FE: numEvents %d, receivedEvents %d\n", numEvents, numRecvEvents);
  assert(numRecvEvents == numEvents);
}

// BaseStats - MRNet has built-in stats filters now: 
//    TFILTER_MIN, TFILTER_MAX,
//    TFILTER_SUM and TFILTER_AVG
// However, without specialize filters, we cannot compute the variance
//    and standard deviations.
void protocolBaseStats() {
  // set up the appropriate streams. ToM will respond using the streams
  //   it receives.
  time_aggregate = 0.0;
  double start_aggregate = ToM_getTimeOfDay();
  double end_aggregate;

  Stream *statStream;
  Stream *nameStream;

  int tag;
  PacketPtr p;

  double *sums;
  int num_sums = 0;
  double *sumofsqr;
  int num_sumofsqr = 0;
  double *mins;
  int num_mins = 0;
  double *maxes;
  int num_maxes = 0;
  int *numContrib;
  int num_contrib_len = 0;

  double *std_devs;
  double *contrib_means;
  double *contrib_std_devs;

  int totalThreads = 0;

  // Ask for the names first
  int threadId = -1;
  int numFunc = 0;
  numMetrics = 0;
  nameStream = net->new_Stream(comm_BC, baseStatsNameFilterId);
  STREAM_FLUSHSEND(nameStream, PROT_BASESTATS, "%d", PROT_BASESTATS);
  nameStream->recv(&tag, p);
  p->unpack("%d %d %as",
	    &threadId, &numMetrics, &tomNames, &numFunc);
  // sanity checks
  assert(threadId == 0);
  //  printf("Num events = %d\n",numFunc);
  /*
  for (int i=0; i<numFunc; i++) {
    printf("%s\n",funcNames[i]);
  }
  */
  
  // Then Ask for the Stats
  statStream = net->new_Stream(comm_BC, baseStatsFilterId);

  STREAM_FLUSHSEND(statStream, PROT_BASESTATS, "%d", PROT_BASESTATS);
  statStream->recv(&tag, p);
  p->unpack("%d %d %alf %alf %alf %alf %ad %d", 
	    &numEvents, &numCounters,
	    &sums, &num_sums, 
	    &sumofsqr, &num_sumofsqr,
	    &mins, &num_mins, 
	    &maxes, &num_maxes,
	    &numContrib, &num_contrib_len,
	    &totalThreads);

  means = new double[numEvents*numCounters*TOM_NUM_VALUES];
  std_devs = new double[numEvents*numCounters*TOM_NUM_VALUES];
  contrib_means = new double[numEvents*numCounters*TOM_NUM_VALUES];
  contrib_std_devs = new double[numEvents*numCounters*TOM_NUM_VALUES];

  /*
  printf("FE: %d %d %d %d %d\n", num_sums, num_sumofsqr, num_mins,
	 num_maxes, num_contrib_len);

  printf("FE: Raw Data %d %d %d\n", numEvents, numCounters, totalThreads);
  for (int evt=0; evt<numEvents; evt++) {
    for (int ctr=0; ctr<numCounters; ctr++) {
      int aIdx = evt*numCounters+ctr;
      printf("[%d,%d] %f %f %f %f %d\n", evt, ctr,
	     sums[aIdx], sumofsqr[aIdx],
	     mins[aIdx], maxes[aIdx], numContrib[evt]);
    }
  }
  */
  for (int evt=0; evt<numEvents; evt++) {
    //printf("FE: [event %d]\n", evt);
    //    printf("FE: [%s]\n", eventNames[evt]);
    for (int ctr=0; ctr<numCounters; ctr++) {
      for (int i=0; i<TOM_NUM_VALUES; i++) {
	int aIdx = 
	  evt*numCounters*TOM_NUM_VALUES+
	  ctr*TOM_NUM_VALUES+
	  i;
	
	// Compute derived statistics.
	means[aIdx] = sums[aIdx]/totalThreads;
	contrib_means[aIdx] = sums[aIdx]/numContrib[evt];
	std_devs[aIdx] = 
	  sqrt((sumofsqr[aIdx]/totalThreads) - 
	       (((2*means[aIdx])/totalThreads)*sums[aIdx]) +
	       (means[aIdx]*means[aIdx]));
	contrib_std_devs[aIdx] =
	  sqrt((sumofsqr[aIdx]/numContrib[evt]) - 
	       (((2*contrib_means[aIdx])/numContrib[evt])*sums[aIdx]) +
	       (contrib_means[aIdx]*contrib_means[aIdx]));
	/*
	  printf("FE: Counter %d\n", ctr);
	  printf("FE: mean:%f stddev:%f cmean:%f cstddev:%f\n",
	  means[aIdx], std_devs[aIdx],
	  contrib_means[aIdx], contrib_std_devs[aIdx]);
	  printf("    sum:%f min:%f max:%f\n",
	  sums[aIdx], mins[aIdx], maxes[aIdx]);
	  fflush(stdout);
	*/
      }
    }
  }

  // Option to send the data back to the clients. There are occasions
  //   where the results of one operation is of use to another.
  // For example, the min and max values can be re-used for histogram
  //   binning. The mean and standard deviation values can be used to
  //   filter unimportant events from histogramming and clustering
  //   operations.
  if (broadcastResults) {
    int numValues = numEvents*numCounters;
    printf("FE: Broadcasting %d results back to BE\n", numValues);
    STREAM_FLUSHSEND(statStream, PROT_BASESTATS,
		     "%alf %alf %alf %alf", 
		     means, numValues,
		     std_devs, numValues,
		     mins, numValues,
		     maxes, numValues);
  }
  end_aggregate = ToM_getTimeOfDay();
  time_aggregate = end_aggregate - start_aggregate;
  // Protocol effectively over at this point.

  // *CWL* - Consider a modular way of transfering the derived data out 
  //         from the front-end:
  //   1. an external client
  //   2. file output
  // Profile format output
  char profileName[512];
  char profileNameTmp[512];
  char aggregateMeta[512];
  sprintf(profileNameTmp, "%s/.temp.mean.%d.0.0",profiledir,
	  invocationIndex);
  sprintf(profileName, "%s/mean.%d.0.0",profiledir, invocationIndex);
  sprintf(aggregateMeta,"<attribute><name>%s</name><value>%.4G seconds</value></attribute>",
	  "Mean Aggregation Time",time_aggregate/1000000.0f);
  FILE *profile = fopen(profileNameTmp,"w");
  // *CWL* - templated_functions_MULTI_<metric name> should be the            
  //         general output format. (See TauCollate.cpp).
  fprintf(profile, "%d templated_functions_MULTI_TIME\n", numEvents);
  fprintf(profile, "# Name Calls Subrs Excl Incl ProfileCalls % <metadata><attribute><name>TAU Monitoring Transport</name><value>MRNet</value></attribute>%s</metadata>\n",
	   aggregateMeta);
  for (int m=0; m<numCounters; m++) {
    for (int f=0; f<numEvents; f++) {
      int aIdx = f*numCounters*TOM_NUM_VALUES + m*TOM_NUM_VALUES;
      // *CWL* use a hard-code for now. Proper solution is to loop through
      //    the data types (which seems like an overkill).
      fprintf(profile, 
	      "\"%s\" %.16G %.16G %.16G %.16G 0 GROUP=\"TAU_DEFAULT\"\n", 
	      tomNames[f+numCounters], 
	      means[aIdx+TOM_VAL_CALL], 
	      means[aIdx+TOM_VAL_SUBR], 
	      means[aIdx+TOM_VAL_EXCL], 
	      means[aIdx+TOM_VAL_INCL]);
    }
    // where there is more than 1 metric (TIME), we will write multiple
    //   blocks.
  }
  fprintf(profile, "0 aggregates\n");
  fclose(profile);
  rename(profileNameTmp, profileName);
}

void protocolHistogram() {
  Stream *histStream;

  int tag;
  PacketPtr p;
  double max;
  double min;

  // Get start timestamp here.
  time_hist = 0.0;
  double start_hist = ToM_getTimeOfDay();
  double end_hist;

  // *CWL* temporary hardcode of 0.01%
  double threshold = 0.01; 
  // *CWL* temporary hardcode of 20 bins
  int numBins = 20;

  int numDataItems;
  int *histBins;

  int keepItem[numEvents*numCounters];
  int numKeep = 0;

  histStream = net->new_Stream(comm_BC, histogramFilterId);

  for (int ctr=0; ctr<numCounters; ctr++) {
    double totalMeans = 0.0;
    // printf("FE: Histogram calculations for Counter %d\n", ctr);
    for (int evt=0; evt<numEvents; evt++) {
      totalMeans += means[evt*numCounters+ctr];
    }
    for (int evt=0; evt<numEvents; evt++) {
      int aIdx = evt*numCounters+ctr;
      // printf("FE: [%s]\n", eventNames[evt]);
      // printf("FE: [event %d]\n", evt);
      // printf("FE: Total:%f Event:%f Prop:%f\n",
      //     totalMeans, means[aIdx], means[aIdx]/totalMeans);
      keepItem[aIdx] = (means[aIdx]/totalMeans)<threshold?0:1;
      numKeep += keepItem[aIdx];
    }
  }
  // printf("FE: Sending results to back-ends for binning.\n");

  STREAM_FLUSHSEND(histStream, PROT_HIST, "%ad %d %d", 
		   keepItem, numEvents*numCounters, numKeep, numBins);
  histStream->recv(&tag, p);
  p->unpack("%ad", &histBins, &numDataItems);

  // Get end timestamp here and calculate time_hist.
  end_hist = ToM_getTimeOfDay();
  time_hist = end_hist - start_hist;
  printf("FE: Histogramming took %.4G seconds\n", time_hist/1000000.0f);

  // Output. Modularize eventually.
  //   Options can include output to file.
  /*
  printf("FE: Histograms Bins *** \n");
  int histIdx = 0;
  for (int evt=0; evt<numEvents; evt++) {
    //    printf("FE: [%s]\n", eventNames[evt]);
    printf("FE: [event %d]\n", evt);
    for (int ctr=0; ctr<numCounters; ctr++) {
      if (keepItem[evt*numCounters+ctr] == 1) {
	printf("FE: Ctr %d | ", ctr);
	for (int bin=0; bin<numBins; bin++) {
	  printf("%d ",histBins[histIdx*numBins+bin]);
	}
	histIdx++;
      }
    }
  }
  printf("\n");
  */
  FILE *histoFile;
  char histFileNameTmp[512];
  char histFileName[512];
  sprintf (histFileName, "%s/tau.histograms.%d", profiledir,
	   invocationIndex);
  sprintf (histFileNameTmp, "%s/.temp.tau.histograms.%d", profiledir,
	   invocationIndex);
  int numGlobalCounters = 1;
  histoFile = fopen(histFileNameTmp, "w");
  fprintf (histoFile, "%d\n", numEvents);
  fprintf (histoFile, "%d\n", numGlobalCounters);
  fprintf (histoFile, "%d\n", numBins);
  for (int i=0; i<numGlobalCounters; i++) {
    // this should be for each metric type.
    fprintf (histoFile, "Exclusive %s\n", "TIME");
    fprintf (histoFile, "Inclusive %s\n", "TIME");
  }
  fprintf (histoFile, "Number of calls\n");
  fprintf (histoFile, "Child calls\n");
  
  char **funcNames = &tomNames[numMetrics];

  for (int e=0; e<numEvents; e++) {
    fprintf(histoFile, "%s\n", funcNames[e]);
  }
  fclose(histoFile);
  rename(histFileNameTmp, histFileName);
}

double ToM_getTimeOfDay() {
  double timestamp;

  struct timeval tp;
  gettimeofday(&tp, 0);
  timestamp = 
    (double)(((unsigned long long)tp.tv_sec * 
	      (unsigned long long)1e6 + 
	      (unsigned long long)tp.tv_usec)*1.0);
  return timestamp;
}
