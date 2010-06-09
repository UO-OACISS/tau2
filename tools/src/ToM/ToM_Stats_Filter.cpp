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
  const char *ToM_Stats_Filter_format_string = "%alf %alf %alf %alf %ad %ad";

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
    double sum = 0.0;
    double sumOfSquares = 0.0;
    double min;
    double max;
    int num_contrib = 0;
    int num_threads = 0;

    int numEvt, numCtr;

    bool firstNonNull = true;

    for (int i=0; i<pin.size(); i++) {
      PacketPtr curr = pin[i];

      double *in_sum;
      int in_sum_len;
      double *in_sos;
      int in_sos_len;
      double *in_min;
      int in_min_len;
      double *in_max;
      int in_max_len;
      int numData;
      int in_numThreads;

      curr->unpack(ToM_Stats_Filter_format_string,
		   &in_sum, &in_sum_len, &in_sos, &in_sos_len,
		   &in_min, &in_min_len, &in_max, &in_max_len,
		   &numData, &in_numThreads);

      //      printf("Node: Received [p:%d] - %f %f %f %f %d %d\n", i,
      //	     in_sum, in_sos, in_min, in_max, numData, in_numThreads);

      // handling cases where a child sends a null-contribution
      if (numData > 0) {
	if (firstNonNull) {
	  min = in_min;
	  max = in_max;
	  firstNonNull = false;
	} else {
	  if (min > in_min) {
	    min = in_min;
	  }
	  if (max < in_max) {
	    max = in_max;
	  }
	}
	sum += in_sum;
	sumOfSquares += in_sos;
	num_contrib += numData;
      }
      // always take care of all threads present
      num_threads += in_numThreads;
    }

    PacketPtr new_packet (new Packet(pin[0]->get_StreamId(),
				     pin[0]->get_Tag(),
				     ToM_Stats_Filter_format_string, 
				     sum, sumOfSquares, min, max,
				     num_contrib, num_threads));
    pout.push_back(new_packet);
  }

}
