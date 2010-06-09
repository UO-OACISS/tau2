#include "Profile/TauMonMrnet.h"

#include "mrnet/Packet.h"
#include "mrnet/NetworkTopology.h"

using namespace MRN;

extern "C" {

  // Experimental filter that will take only single statistics data sets
  const char *ToM_StatsSingle_Filter_format_string = "%lf %lf %lf %lf %d %d";

  // Get Sum, Sum of Squares, Count, Min and Max
  //    - Avg, Variance, Std Dev can be derived from these values.
  void ToM_StatsSingle_Filter(std::vector<PacketPtr>& pin,
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

    bool firstNonNull = true;

    for (int i=0; i<pin.size(); i++) {
      PacketPtr curr = pin[i];

      double in_sum;
      double in_sos;
      double in_min;
      double in_max;
      int numData;
      int in_numThreads;

      curr->unpack(ToM_StatsSingle_Filter_format_string,
		   &in_sum, &in_sos,
		   &in_min, &in_max,
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
				     ToM_StatsSingle_Filter_format_string, 
				     sum, sumOfSquares, min, max,
				     num_contrib, num_threads));
    pout.push_back(new_packet);
  }

}
