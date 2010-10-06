#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <sys/time.h>
#include <sys/stat.h>

#include <stdint.h>
#include <string.h>

#include "mrnet/MRNet.h"
#include "Profile/TauMonMrnet.h"

using namespace MRN;
using namespace std;

Network *net = NULL;
Communicator *comm_BC;

Stream *ctrl_stream;

int num_callbacks;
int num_backends;

int unifyFilterId;
int baseStatsFilterId;
int baseStatsNameFilterId;
int histogramFilterId;
int clusterFilterId;

bool broadcastResults;

// *CWL* General purpose filters?
int syncFilterId;

char *profiledir;

// Shared information between Base Stats and Histogram protocols
// *CWL* - find a better way of implementing this, probably through
//    object-based interfaces. 
int numEvents, numCounters;
double *means;
double *mins;
double *maxes;
double *std_devs;

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
double time_cluster;

double ToM_getTimeOfDay();
bool vectorModified(double *changeVector, int numK, int numEvents);

void controlLoop();

// early idea for eventual modularization
void registerProtocols();
void registerUnify();
void registerBaseStats();
void registerHistogram();
void registerClustering();

void monitoringProtocol();
void builtInProtocols();
void protocolUnify();
void protocolBaseStats();
void protocolHistogram();
void protocolClustering();

void BE_Add_Callback( Event* evt, void* )
{
  if ((evt->get_Class() == Event::TOPOLOGY_EVENT) &&
      (evt->get_Type() == TopologyEvent::TOPOL_ADD_BE))
    num_callbacks++;
}

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
    num_backends = atoi(argv[2]);

    // *CWL* can make this a switch next time.
    broadcastResults = true;

    printf("FE: Creating network\n");
    
    // If backend_exe (2nd arg) and backend_args (3rd arg) are both NULL,
    // then all nodes specified in the topology are internal tree nodes.
    net = Network::CreateNetworkFE(topology_file, NULL, NULL);
    bool cbOK = net->register_EventCallback(Event::TOPOLOGY_EVENT,
					    TopologyEvent::TOPOL_ADD_BE,
					    BE_Add_Callback, NULL);
    if (cbOK == false) {
      fprintf(stdout, "Failed to register callback for back-end add topology event\n");
      delete net;
      return -1;
    }
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
    printf("FE: Waiting for %u backends to connect.\n", num_backends);
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

    do {
      sleep(1);
    } while (num_callbacks != num_backends);
    printf("FE: All application backends connected!\n");

    // Specialized stream construction
    // Broadcast stream
    comm_BC = net->get_BroadcastCommunicator();
    ctrl_stream = net->new_Stream(comm_BC, syncFilterId);

    // should backends go away?
    // net->set_TerminateBackEndsOnShutdown(true);

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

      // This is hackish. Give the backends some time to successfully
      //   complete the call to waitfor_Shutdown before a network
      //   delete. This sleep can probabaly go away unless there are
      //   race conditions that are not handled correctly by MRNet.
      sleep(1);
      printf("FE: Tearing down MRNet network.\n");
      delete net;
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
  registerClustering();
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

void registerClustering() {
  // set up pseudo-random seed
  srand(11337);
  clusterFilterId = net->load_FilterFunc("ToM_Cluster_Filter.so",
					 "ToM_Cluster_Filter");
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
  protocolClustering();
}

void protocolUnify() {
  int tag;
  PacketPtr p;
  
  // ask application for name strings. These can be acquired from
  //   Rank 0 after MPI-based unification.
  int numRecvEvents;
  // p->unpack("%as", &eventNames, &numRecvEvents);
  //  printf("FE: numEvents %d, receivedEvents %d\n", numEvents, numRecvEvents);
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

  Stream *nameStream;
  Stream *statStream;

  int tag;
  PacketPtr p;

  // intermediate data items
  double *sums;
  double *sumofsqr;

  int num_sums = 0;
  int num_sumofsqr = 0;
  int num_mins = 0;
  int num_maxes = 0;
  
  int totalThreads = 0;

  // Ask for the event and metric names
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

  // Ask for the Stats
  statStream = net->new_Stream(comm_BC, baseStatsFilterId);
  STREAM_FLUSHSEND(statStream, PROT_BASESTATS, "%d", PROT_BASESTATS);
  statStream->recv(&tag, p);
  p->unpack("%d %d %alf %alf %alf %alf %d", 
	    &numEvents, &numCounters,
	    &sums, &num_sums, 
	    &sumofsqr, &num_sumofsqr,
	    &mins, &num_mins, 
	    &maxes, &num_maxes,
	    &totalThreads);

  int numItems = numCounters*TOM_NUM_CTR_VAL+TOM_NUM_FUN_VAL;
  means = new double[numEvents*numItems];
  std_devs = new double[numEvents*numItems];

  char **funcNames = &tomNames[numMetrics];
  for (int evt=0; evt<numEvents; evt++) {
    for (int itm=0; itm<numItems; itm++) {
      int aIdx = evt*numItems + itm;

      // Compute derived statistics.
      means[aIdx] = sums[aIdx]/totalThreads;
      std_devs[aIdx] = 
	sqrt((sumofsqr[aIdx]/totalThreads) - 
	     (((2*means[aIdx])/totalThreads)*sums[aIdx]) +
	     (means[aIdx]*means[aIdx]));
    }
  }

  // Option to send the data back to the clients. There are occasions
  //   where the results of one operation is of use to another.
  // For example, the min and max values can be re-used for histogram
  //   binning. The mean and standard deviation values can be used to
  //   filter unimportant events from histogramming and clustering
  //   operations.
  if (broadcastResults) {
    int numValues = numEvents*numItems;
    STREAM_FLUSHSEND(statStream, PROT_BASESTATS,
		     "%alf %alf %alf %alf", 
		     means, numValues,
		     std_devs, numValues,
		     mins, numValues,
		     maxes, numValues);
  }
  end_aggregate = ToM_getTimeOfDay();
  time_aggregate = end_aggregate - start_aggregate;

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
  // *CWL* Output will ignore counters altogether. Profiles are an
  //    inappropriate format for the output with multiple counters.
  //  for (int m=0; m<numCounters; m++) {
    for (int f=0; f<numEvents; f++) {
      int aIdx = f*numItems;
      //      int aIdx = f*numCounters*TOM_NUM_VALUES + m*TOM_NUM_VALUES;
      // *CWL* use a hard-code for now. Proper solution is to loop through
      //    the data types (which seems like an overkill).
      fprintf(profile, 
	      "\"%s\" %.16G %.16G", tomNames[f+numCounters], 
	      means[aIdx+TOM_FUN_CALL], 
	      means[aIdx+TOM_FUN_SUBR]);
      aIdx = aIdx + TOM_NUM_FUN_VAL;
      fprintf(profile,
	      " %.16G %.16G 0 GROUP=\"TAU_DEFAULT\"\n", 
	      means[aIdx+TOM_CTR_EXCL], 
	      means[aIdx+TOM_CTR_INCL]);
    }
  // }
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

  histStream = net->new_Stream(comm_BC, histogramFilterId);

  // *CWL* temporary hardcode for 20 bins
  int numBins = 20;
  //  printf("FE: Instructing back-ends to bin with %d bins.\n",numBins);
  STREAM_FLUSHSEND(histStream, PROT_HIST, "%d", numBins);

  int numDataItems;
  int *histBins;
  histStream->recv(&tag, p);
  p->unpack("%ad", &histBins, &numDataItems);

  // Get end timestamp here and calculate time_hist.
  end_hist = ToM_getTimeOfDay();
  time_hist = end_hist - start_hist;
  printf("FE: Histogramming took %.4G seconds\n", time_hist/1000000.0f);

  FILE *histoFile;
  char histFileNameTmp[512];
  char histFileName[512];
  sprintf (histFileName, "%s/tau.histograms.%d", profiledir,
	   invocationIndex);
  sprintf (histFileNameTmp, "%s/.temp.tau.histograms.%d", profiledir,
	   invocationIndex);
  int numHistogramsPerEvent = numCounters*TOM_NUM_CTR_VAL+TOM_NUM_FUN_VAL;
  histoFile = fopen(histFileNameTmp, "w");
  fprintf (histoFile, "%d\n", numEvents);
  fprintf (histoFile, "%d\n", numHistogramsPerEvent);
  fprintf (histoFile, "%d\n", numBins);
  for (int i=0; i<numCounters; i++) {
    fprintf (histoFile, "Exclusive %s\n", tomNames[i]);
    fprintf (histoFile, "Inclusive %s\n", tomNames[i]);
  }
  fprintf (histoFile, "Number of calls\n");
  fprintf (histoFile, "Child calls\n");
  
  char **funcNames = &tomNames[numMetrics];

  for (int e=0; e<numEvents; e++) {
    fprintf(histoFile, "%s\n", funcNames[e]);
    int dataBaseOffset = e*numHistogramsPerEvent;
    for (int m=0; m<numCounters; m++) {
      int ctrIdx = dataBaseOffset + TOM_NUM_FUN_VAL + m*TOM_NUM_CTR_VAL;
      for (int type=0; type<TOM_NUM_CTR_VAL; type++) {
	int finalCtrIdx = ctrIdx + type;
	int histIdx = finalCtrIdx*numBins;
	fprintf(histoFile, "%.16G %.16G ", 
		mins[finalCtrIdx], maxes[finalCtrIdx]);
	for (int b=0; b<numBins; b++) {
	  fprintf(histoFile, "%d ", 
		  histBins[histIdx + b]);
	}
	fprintf(histoFile, "\n");
      }
    }
    for (int type=0; type<TOM_NUM_FUN_VAL; type++) {
      int ctrIdx = dataBaseOffset + type;
      int histIdx = ctrIdx*numBins;
      fprintf(histoFile, "%.16G %.16G ", 
	      mins[ctrIdx], maxes[ctrIdx]);
      for (int b=0; b<numBins; b++) {
	fprintf(histoFile, "%d ", histBins[histIdx + b]);
      }
      fprintf(histoFile, "\n");
    }
  }
  fclose(histoFile);
  rename(histFileNameTmp, histFileName);
}

void protocolClustering() {
  Stream *clusterStream;
  int numK = 5; // default;

  time_cluster = 0.0;
  double start_cluster = ToM_getTimeOfDay();
  double end_cluster;

  char *numKString = getenv("TOM_CLUSTER_K");
  if (numKString != NULL) {
    numK = atoi(numKString); // user specification
  }
  if (numK > num_backends) {
    numK = num_backends;
    printf("FE: Warning - K larger than number of backends %d. Set to %d\n", 
	   num_backends, numK);
  }
  printf("FE: Start Clustering with K=%d\n",numK);

  // We are clustering across functions only. It might
  //   be useful in the future to cluster across the 
  //   (function X counter) domain. In those cases, we'll need a
  //   non-trivial normalization function.

  // For this version, work with Exclusive TIME alone.
  //  int numItemsPerEvent = (numCounters*TOM_NUM_CTR_VAL)+TOM_NUM_FUN_VAL;
  int numItemsPerEvent = 1;
  int numItemsPerK = numEvents*numItemsPerEvent;
  int dataLength = numK*numItemsPerK;

  double *clusterCentroids;
  double *clusterCentroidVectors;
  int *clusterNumMembers;

  // set up the clustering stream 
  clusterStream = net->new_Stream(comm_BC, clusterFilterId);

  // activate backends with value of k. Note the use of the
  //   control stream instead of the clustering stream to allow
  //   a simple acknowledgement from the backends.
  STREAM_FLUSHSEND(ctrl_stream, PROT_CLUST_KMEANS, "%d", numK);

  // receive acknowledgement. The value is not important but
  //   could be used for sanity checks.
  int ackVal;
  int tag;
  PacketPtr p;
  ctrl_stream->recv(&tag, p);
  p->unpack("%d", &ackVal);

  // *CWL* Do not muck up the initial implementation with Normalization
  //   issues until they become necessary!
  printf("FE: Backends acknowledged Clustering operation\n");

  map<int,int> nodeHash; // assume thread 0 will represent a node
  map<int,int>::iterator it;
  // Randomly choose initial k nodes as centroids. We will be lazy here
  //   and have the same k nodes represent the centroids for ALL data
  //   types.
  // Efficiency also dictates that we numK is small relative to the
  //   of nodes. We can work in code to handle that condition later.
  int choiceCount = 0;
  int *choices = new int[numK];
  while (choiceCount < numK) {
    int choice = (int)(floor((rand()*1.0/RAND_MAX)*num_backends));
    // printf("Picking %d\n", choice);
    // paranoia
    assert((choice >= 0) && (choice < num_backends));
    if (nodeHash.count(choice) > 0) {
      continue;
    } else {
      choices[choiceCount] = choice;
      nodeHash[choice] = 1;
      choiceCount++;
    }
  }

  printf("FE: Random initial centroids determined.\n");
  /*
  printf("[");
  for (int i=0; i<numK; i++) {
    printf("%d ", choices[i]);
  }
  printf("]\n");
  */
  // broadcast the choices and receive the initial vectors
  //   from the designated participants. Results can return in the
  //   form of the standard change vector, but interpreted differently.
  STREAM_FLUSHSEND(clusterStream, PROT_CLUST_KMEANS, "%ad",
		   choices, numK);
  int centroidDataLength;
  int numMember;
  clusterStream->recv(&tag, p);
  p->unpack("%alf %ad", 
	    &clusterCentroidVectors, &centroidDataLength,
	    &clusterNumMembers, &numMember);

  // printf("FE: Received %d Initial Centroids\n", centroidDataLength);
  /*
  for (int i=0; i<centroidDataLength; i++) {
    printf("%.16G ", clusterCentroidVectors[i]);
  }
  printf("\n");
  */

  // broadcast the initial cluster centroids reported by participants
  //   to everyone. These initial centroids have exactly 1 member, so
  //   there is no need for the vectors to be converted into actual
  //   centroids.
  STREAM_FLUSHSEND(clusterStream, PROT_CLUST_KMEANS, "%alf",
		   clusterCentroidVectors, centroidDataLength);
  
  // Loop until done.
  double *changeVector; // congruent to centroid structure
  int *changeNumMembers; // congruent to clusterNumMembers
  int changeVectorDataLength;
  int numChangeNumMembers;

  int iterationCount = 0;
  int stop = 0;
  clusterCentroids = new double[numK*numItemsPerK];
  do {
    // receive a centroid modification vector.
    clusterStream->recv(&tag, p);
    p->unpack("%alf %ad", 
	      &changeVector, &changeVectorDataLength, 
	      &changeNumMembers, &numChangeNumMembers);

    //    printf("FE: Cluster Iteration %d, received new change vector\n",
    //	   iterationCount);
    /*
    for (int i=0; i<changeVectorDataLength; i++) {
      printf("%.16G ",changeVector[i]);
    }
    printf("\n");
    for (int i=0; i<numChangeNumMembers; i++) {
      printf("%d ",changeNumMembers[i]);
    }
    printf("\n");
    */
    if (!vectorModified(changeVector, numK, numEvents)) {
      stop = 1;
      printf("FE: Informing Backends convergence attained after %d steps\n",
	     iterationCount+1);
      STREAM_FLUSHSEND(ctrl_stream, PROT_CLUST_KMEANS, "%d",
		       stop);    
      ctrl_stream->recv(&tag, p);
      p->unpack("%d", &ackVal);
      // printf("FE: Backends acknowledged Convergence\n");
      break;
    }

    // update the centroid vectors and centroid membership
    for (int k=0; k<numK; k++) {
      // handling special cases
      bool vacateK = false;
      bool populateK = false;
      if ((clusterNumMembers[k] == -changeNumMembers[k]) &&
	  (changeNumMembers[k] < 0)) {
	// printf("FE: vacating %d\n", k);
	vacateK = true;
      }
      if ((clusterNumMembers[k] == 0) &&
	  (changeNumMembers[k] > 0)) {
	// printf("FE: populating %d\n", k);
	populateK = true;
      }
      clusterNumMembers[k] += changeNumMembers[k];
      assert(clusterNumMembers[k] >= 0);
      for (int evt=0; evt<numEvents; evt++) {
	int vIdx = k*numEvents+evt;
	if (vacateK) {
	  // retain old values, so the centroid does not move. But
	  //   convert it into a point using old membership.
	  clusterCentroidVectors[vIdx] /= -changeNumMembers[k];
	} else if (populateK) {
	  // take the value of the change vector, do not accumulate.
	  clusterCentroidVectors[vIdx] = changeVector[vIdx];
	} else {
	  clusterCentroidVectors[vIdx] += changeVector[vIdx];
	}
	if (clusterNumMembers[k] > 0) {
	  // convert vector into point
	  clusterCentroids[vIdx] = 
	    clusterCentroidVectors[vIdx]/clusterNumMembers[k];
	} else {
	  // Empty cluster.
	  // send the old vacated position. Note that in this case,
	  //   the original vector has already been converted into a point.
	  clusterCentroids[vIdx] = clusterCentroidVectors[vIdx];
	}
      }
    }

    // printf("FE: Informing Backends no convergence\n");
    STREAM_FLUSHSEND(ctrl_stream, PROT_CLUST_KMEANS, "%d",
		     stop);
    ctrl_stream->recv(&tag, p);
    p->unpack("%d", &ackVal);
    // printf("FE: Backends acknowledged No Convergence\n");

    iterationCount++;

    // printf("FE: Broadcasting updated centroids\n");
    //   - broadcast updated centroids.
    STREAM_FLUSHSEND(clusterStream, PROT_CLUST_KMEANS, "%alf",
		     clusterCentroids, numK*numItemsPerK);
    // Stop when modification vector is zero.
  } while (true);

  // Get end timestamp here and calculate time_hist.
  end_cluster = ToM_getTimeOfDay();
  time_cluster = end_cluster - start_cluster;
  printf("FE: Clustering took %.4G seconds\n", time_cluster/1000000.0f);

  // output profile fakery. K profiles are written per frame. The
  //   frame number is captured in the filename.
  FILE *clusterFile;
  char clusterDirName[512];
  char clusterFileNameTmp[512];
  char clusterFileName[512];
  char clusterMeta[4096];
  sprintf(clusterDirName, "%s/cluster_%d",profiledir, invocationIndex);
  mkdir(clusterDirName,0755);
  for (int k=0; k<numK; k++) {
    sprintf(clusterFileNameTmp, "%s/.temp.profile.%d.0.0",clusterDirName,k);
    sprintf(clusterFileName, "%s/profile.%d.0.0",clusterDirName,k);
    sprintf(clusterMeta,"<attribute><name>%s</name><value>%.4G seconds</value></attribute><attribute><name>%s</name><value>%d</value></attribute><attribute><name>%s</name><value>%d</value></attribute>",
	    "Clustering Time",time_cluster/1000000.0f,
	    "cluster-membership", clusterNumMembers[k],
	    "Clustering Convergence Steps", iterationCount);
    FILE *clusterFile = fopen(clusterFileNameTmp,"w");
    fprintf(clusterFile, "%d templated_functions_MULTI_TIME\n", numEvents);
    fprintf(clusterFile, "# Name Calls Subrs Excl Incl ProfileCalls % <metadata><attribute><name>TAU Monitoring Transport</name><value>MRNet</value></attribute>%s</metadata>\n",
	    clusterMeta);
    for (int f=0; f<numEvents; f++) {
      int aIdx = k*numEvents+f;
      // 1 CALL and no SUBR faked. INCL faked to be EXCL.
      fprintf(clusterFile, 
	      "\"%s\" %.16G %.16G %.16G %.16G 0 GROUP=\"TAU_DEFAULT\"\n", 
	      tomNames[f+numCounters], 
	      1.0, 0.0, clusterCentroids[aIdx], clusterCentroids[aIdx]);  
    }
    fprintf(clusterFile, "0 aggregates\n");
    fclose(clusterFile);
    rename(clusterFileNameTmp, clusterFileName);
  }
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

bool vectorModified(double *changeVector, int numK, int numEvents) {
  for (int k=0; k<numK; k++) {
    for (int evt=0; evt<numEvents; evt++) {
      if (changeVector[k*numEvents+evt] > 0.0) {
	return true;
      }
    }
  }
  return false;
}
