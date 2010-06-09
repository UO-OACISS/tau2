/****************************************************************************
 * Copyright © 2003-2009 Dorian C. Arnold, Philip C. Roth, Barton P. Miller *
 *                  Detailed MRNet usage rights in "LICENSE" file.          *
 ****************************************************************************/

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <stdint.h>

#include "mrnet/MRNet.h"
#include "Profile/TauMonMrnet.h"

using namespace MRN;
using namespace std;

Network *net = NULL;
Communicator *comm_BC;

Stream *ctrl_stream;

int baseStatsFilterId;
int histogramFilterId;

bool broadcastResults;

// *CWL* General purpose filters?
int syncFilterId;

// Shared information between Base Stats and Histogram protocols
// *CWL* - find a better way of implementing this, probably through
//    object-based interfaces. 
int numEvents, numCounters;
double *means;

void controlLoop();

// early idea for eventual modularization
void registerProtocols();
void registerBaseStats();
void registerHistogram();

void monitoringProtocol();
void protocolBaseStats();
void protocolHistogram();

void write_be_connections(vector<NetworkTopology::Node *>& leaves, 
			  int num_net_nodes,
			  int num_be)
{
   FILE *f;
   const char* connfile = "./attachBE_connections";
   if ((f = fopen(connfile, (const char *)"w+")) == NULL)
   {
      perror("fopen");
      exit(-1);
   }

   // block-assignment of back-ends to leaf nodes
   int num_leaves = leaves.size();
   int be_per_leaf = (int)(ceil((double)num_be/(double)num_leaves));
   int remaining_be = num_be;
   for (int i=0; i<num_leaves; i++) {
     int be_to_leaf = (remaining_be >= be_per_leaf)?be_per_leaf:remaining_be;
     for (int be_offset=0; be_offset<be_to_leaf; be_offset++) {
       int be_real_rank = num_be - remaining_be + be_offset;
       int be_mrnet_rank = be_real_rank + num_net_nodes;
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
    char* topology_file = argv[1];
    int num_backends = atoi(argv[2]);

    // *CWL* can make this a switch next time.
    broadcastResults = true;

    printf("Creating network\n");
    
    // If backend_exe (2nd arg) and backend_args (3rd arg) are both NULL,
    // then all nodes specified in the topology are internal tree nodes.
    net = Network::CreateNetworkFE(topology_file, NULL, NULL);

    printf("Network created\n");

    // Load filter functions now (need more elegant way later)
    registerProtocols();
    net->load_FilterFunc("ToM_Test_Filter.so","ToM_Test_Filter");
    syncFilterId = net->load_FilterFunc("ToM_Sync_Filter.so",
					"ToM_Sync_Filter");
  
    // Query net for topology object
    NetworkTopology * topology = net->get_NetworkTopology();
    num_mrnet_nodes = topology->get_NumNodes();
    vector< NetworkTopology::Node * > internal_leaves;
    topology->get_Leaves(internal_leaves);

    // Write connection information to temporary file
    write_be_connections(internal_leaves, num_mrnet_nodes, num_backends);

    fprintf(stdout, "MRNet network successfully created.\n");
    fprintf(stdout, "Waiting for %u backends to connect.\n", num_backends );
    fflush(stdout);

    set<NetworkTopology::Node *> be_nodes;
    do {
        sleep(1);
	topology->get_BackEndNodes(be_nodes);
    } while (be_nodes.size() < num_backends);
    fprintf(stdout, "ToM_FE: All application backends connected!\n");

    comm_BC = net->get_BroadcastCommunicator();
    ctrl_stream = net->new_Stream(comm_BC, syncFilterId);

    // should backends go away?
    net->set_TerminateBackEndsOnShutdown(false);

    fprintf(stdout, "Establish control streams to back-ends\n");
    /*
    if ((ctrl_stream->send(TOM_CONTROL, "%d", 1337) == -1) ||
	(ctrl_stream->flush() == -1)) {
      printf("stream::send(%d) failure\n", 1337);
      return -1;
    }
    */
    STREAM_FLUSHSEND(ctrl_stream, TOM_CONTROL, "%d", broadcastResults?1:0);

    // Testing number of recvs the FE should expect here.
    /* DEBUG ONLY
    while (true) {
      int tag;
      PacketPtr p;
      int protocolTag;
     
      printf("Waiting\n");
      ctrl_stream->recv(&tag, p);
      p->unpack("%d", &protocolTag);
      printf("Got %d\n", protocolTag);
    }
    */

    // control loop
    controlLoop();

    return 0;
}

void controlLoop() {

  bool processProtocol = true;

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
      printf("FE: Shutting down ToM front-end\n");
      delete net;
      processProtocol = false;
      break;
    }
    case PROT_DATA_READY: {
      printf("FE: Data ready at application backends. Start protocols.\n");
      monitoringProtocol();
      break;
    }
    default: {
      printf("Warning: Unknown protocol tag [%d]\n", protocolTag);
    }
    }
    
  }
}

void registerProtocols() {
  registerBaseStats();
  registerHistogram();
}

void registerBaseStats() {
  baseStatsFilterId = net->load_FilterFunc("ToM_StatsSingle_Filter.so",
					   "ToM_StatsSingle_Filter");
}

void registerHistogram() {
  // load the Histogram filter here.
  histogramFilterId = net->load_FilterFunc("ToM_Histogram_Filter.so",
					   "ToM_Histogram_Filter");
}

void monitoringProtocol() {
  // basically activate all desired protocols on ready signal 
  // from application
  protocolBaseStats();
  protocolHistogram();
  STREAM_FLUSHSEND(ctrl_stream, PROT_EXIT, "%d", PROT_EXIT);
}

// BaseStats - MRNet has built-in stats filters now: 
//    TFILTER_MIN, TFILTER_MAX,
//    TFILTER_SUM and TFILTER_AVG
// However, without specialize filters, we cannot compute the variance
//    and standard deviations.
void protocolBaseStats() {
  // set up the appropriate streams. ToM will respond using the streams
  //   it receives.
  Stream *statCountStream;
  Stream *statStream;

  int tag;
  PacketPtr p;

  statCountStream = net->new_Stream(comm_BC);
  statStream = net->new_Stream(comm_BC, baseStatsFilterId);

  STREAM_FLUSHSEND(statCountStream, PROT_BASESTATS, "%d", PROT_BASESTATS);
		   
  // ask application for number of statistics sets to expect.
  statCountStream->recv(&tag, p);
  p->unpack("%d %d", &numEvents, &numCounters);

  printf("FE: Received %d %d\n", numEvents, numCounters);

  // ask application for static information like name strings etc ...


  // ask application for the necessary waves of data. 
  double *sums;
  double *contrib_means;
  double *std_devs;
  double *contrib_std_devs;
  double *mins;
  double *maxes;

  sums = new double[numEvents*numCounters];
  means = new double[numEvents*numCounters];
  std_devs = new double[numEvents*numCounters];
  contrib_means = new double[numEvents*numCounters];
  contrib_std_devs = new double[numEvents*numCounters];
  mins = new double[numEvents*numCounters];
  maxes = new double[numEvents*numCounters];

  for (int evt=0; evt<numEvents; evt++) {
    for (int ctr=0; ctr<numCounters; ctr++) {
      double sumofsqr;
      int numContrib, numThreads;

      int aIdx = evt*numCounters+ctr;
      STREAM_FLUSHSEND(statStream, PROT_BASESTATS, "%d", PROT_BASESTATS);
      statStream->recv(&tag, p);
      p->unpack("%lf %lf %lf %lf %d %d",
		&sums[aIdx], &sumofsqr, &mins[aIdx], &maxes[aIdx], 
		&numContrib, &numThreads);
      /* DEBUG 
      printf("FE: [%d,%d] Received: %f %f %f %f %d %d\n", evt, ctr,
	     sums[aIdx], sumofsqr, mins[aIdx], maxes[aIdx],
	     numContrib, numThreads);
      */

      // Compute derived statistics.
      means[aIdx] = sums[aIdx]/numThreads;
      contrib_means[aIdx] = sums[aIdx]/numContrib;
      std_devs[aIdx] = 
	sqrt((sumofsqr/numThreads) - 
	     (((2*means[aIdx])/numThreads)*sums[aIdx]) +
	     (means[aIdx]*means[aIdx]));
      contrib_std_devs[aIdx] =
	sqrt((sumofsqr/numContrib) - 
	     (((2*contrib_means[aIdx])/numContrib)*sums[aIdx]) +
	     (contrib_means[aIdx]*contrib_means[aIdx]));
      printf("FE: mean:%f stddev:%f cmean:%f cstddev:%f\n",
	     means[aIdx], std_devs[aIdx],
	     contrib_means[aIdx], contrib_std_devs[aIdx]);
      printf("    sum:%f min:%f max:%f\n",
	     sums[aIdx], mins[aIdx], maxes[aIdx]);
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
    printf("Broadcasting %d results back to BE\n", numValues);
    STREAM_FLUSHSEND(statStream, PROT_BASESTATS,
		     "%alf %alf %alf %alf", 
		     means, numValues,
		     std_devs, numValues,
		     mins, numValues,
		     maxes, numValues);
  }

  // Protocol effectively over at this point.

  // *CWL* - Consider a modular way of transfering the derived data out 
  //         from the front-end:
  //   1. an external client
  //   2. file output
}

void protocolHistogram() {
  Stream *histStream;

  int tag;
  PacketPtr p;
  double max;
  double min;

  // *CWL* temporary hardcode of 0.01%
  double threshold = 0.01; 
  // *CWL* temporary hardcode of 20 bins
  int numBins = 10;

  int numDataItems;
  int *histBins;

  int keepItem[numEvents*numCounters];
  int numKeep = 0;

  histStream = net->new_Stream(comm_BC, histogramFilterId);

  for (int ctr=0; ctr<numCounters; ctr++) {
    double totalMeans = 0.0;
    for (int evt=0; evt<numEvents; evt++) {
      totalMeans += means[evt*numCounters+ctr];
    }
    for (int evt=0; evt<numEvents; evt++) {
      int aIdx = evt*numCounters+ctr;
      printf("[%d,%d]: Total:%f Event:%f Prop:%f\n",
	     evt, ctr, totalMeans, means[aIdx], means[aIdx]/totalMeans);
      keepItem[aIdx] = (means[aIdx]/totalMeans)<threshold?0:1;
      numKeep += keepItem[aIdx];
    }
  }

  STREAM_FLUSHSEND(histStream, PROT_HIST, "%ad %d %d", 
		   keepItem, numEvents*numCounters, numKeep, numBins);
  histStream->recv(&tag, p);
  p->unpack("%ad", &histBins, &numDataItems);

  printf("FE: Received Histograms\n");
  int histIdx = 0;
  for (int evt=0; evt<numEvents; evt++) {
    for (int ctr=0; ctr<numCounters; ctr++) {
      if (keepItem[evt*numCounters+ctr] == 1) {
	printf("[%d,%d]:", evt, ctr);
	for (int bin=0; bin<numBins; bin++) {
	  printf("%d ",histBins[histIdx*numBins+bin]);
	}
	histIdx++;
      }
    }
  }
  printf("\n");
}
