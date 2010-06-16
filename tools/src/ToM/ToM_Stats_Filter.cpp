#include "Profile/TauMonMrnet.h"

#include "mrnet/Packet.h"
#include "mrnet/NetworkTopology.h"

using namespace MRN;

extern "C" {

  // Manage data in complete blocks.
  //   For each function (evt), we know:
  //      1. how many threads contribute to this function
  //      2. There are k counters.
  //   For each counter, we have 1 set of statistics.
  const char *ToM_Stats_Filter_format_string = 
    "%d %d %alf %alf %alf %alf %ad %d";

  // Get Sum, Sum of Squares, Count, Min and Max
  //    - Avg, Variance, Std Dev can be derived from these values.
  void ToM_Stats_Filter(std::vector<PacketPtr>& pin,
			std::vector<PacketPtr>& pout,
			std::vector<PacketPtr>& /* packets_out_reverse */,
			void ** /* client data */,
			PacketPtr& params,
			const TopologyLocalInfo&) {
    // initial values are the same as that of a null-contribution
    //   in the event all of this treenode's children make 
    //   null-contributions.
    double *sums;
    double *sumsofsqr;
    double *mins;
    double *maxes;
    int *activeThreads;
    int totalThreads = 0;

    int numEvents = 0;
    int numCounters = 0; 
    int numItems = 0; // sanity check

    for (int i=0; i<pin.size(); i++) {
      PacketPtr curr = pin[i];

      int in_events = 0;
      int in_counters = 0;

      double *in_sums;
      int in_sums_len = 0;
      double *in_sumsofsqr;
      int in_sumsofsqr_len = 0;
      double *in_mins;
      int in_mins_len = 0;
      double *in_maxes;
      int in_maxes_len= 0;
      int *in_activethreads;
      int in_active_len = 0;
      int in_totalThreads = 0;

      curr->unpack(ToM_Stats_Filter_format_string, &in_events, &in_counters,
		   &in_sums, &in_sums_len, &in_sumsofsqr, &in_sumsofsqr_len,
		   &in_mins, &in_mins_len, &in_maxes, &in_maxes_len,
		   &in_activethreads, &in_active_len, &in_totalThreads);

      // local sanity check
      int in_items = in_events*in_counters;
      assert((in_items == in_sums_len) &&
	     (in_items == in_sumsofsqr_len) &&
	     (in_items == in_mins_len) &&
	     (in_items == in_maxes_len) &&
	     (in_items == in_active_len) &&
	     (in_items > 0));
      
      if (i == 0) {
	sums = in_sums;
	sumsofsqr = in_sumsofsqr;
	mins = in_mins;
	maxes = in_maxes;
	activeThreads = in_activethreads;

	numEvents = in_events;
	numCounters = in_counters;
	numItems = in_items;

	/* DEBUG 
	printf("COMM: Incoming item:\n");
	for (int evt=0; evt<numEvents; evt++) {
	  for (int ctr=0; ctr<numCounters; ctr++) {
	    int aIdx = evt*numCounters+ctr;
	    printf("COMM [%d,%d] %f %f %f %f %d\n", evt, ctr,
		   sums[aIdx], sumsofsqr[aIdx], mins[aIdx], maxes[aIdx],
		   activeThreads[evt]);
	  }
	}
	*/
      } else {
	// global sanity check
	assert((numEvents == in_events) &&
	       (numCounters == in_counters));
	for (int evt=0; evt<numEvents; evt++) {
	  for (int ctr=0; ctr<numCounters; ctr++) {
	    int aIdx = evt*numCounters+ctr;
	    sums[aIdx] += in_sums[aIdx];
	    sumsofsqr[aIdx] += in_sumsofsqr[aIdx];
	    if (mins[aIdx] > in_mins[aIdx]) {
	      mins[aIdx] = in_mins[aIdx];
	    }
	    if (maxes[aIdx] < in_maxes[aIdx]) {
	      maxes[aIdx] = in_maxes[aIdx];
	    }
	  }
	  activeThreads[evt] += in_activethreads[evt];
	}
      }
      totalThreads += in_totalThreads;
    }

    /* DEBUG 
    printf("COMM: Outgoing item:\n");
    for (int evt=0; evt<numEvents; evt++) {
      for (int ctr=0; ctr<numCounters; ctr++) {
	int aIdx = evt*numCounters+ctr;
	printf("COMM [%d,%d] %f %f %f %f %d\n", evt, ctr,
	       sums[aIdx], sumsofsqr[aIdx], mins[aIdx], maxes[aIdx],
	       activeThreads[evt]);
      }
    }
    */

    PacketPtr new_packet (new Packet(pin[0]->get_StreamId(),
				     pin[0]->get_Tag(),
				     ToM_Stats_Filter_format_string, 
				     numEvents, numCounters,
				     sums, numItems,
				     sumsofsqr, numItems,
				     mins, numItems,
				     maxes, numItems,
				     activeThreads, numEvents,
				     totalThreads));

    pout.push_back(new_packet);
  }

}
