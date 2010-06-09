#include "Profile/TauMonMrnet.h"

#include "mrnet/Packet.h"
#include "mrnet/NetworkTopology.h"

using namespace MRN;

extern "C" {

  const char *ToM_Stats_Filter_format_string = "%f %f %f %f %d";

  // Get Sum, Sum of Squares, Count, Min and Max
  //    - Avg, Variance, Std Dev can be derived from these values.
  void ToM_Stats_Filter(std::vector<PacketPtr>& pin,
			std::vector<PacketPtr>& pout,
			std::vector<PacketPtr>& /* packets_out_reverse */,
			void ** /* client data */,
			PacketPtr& params,
			const TopologyLocalInfo&) {
    double sum = 0.0;
    double sumOfSquares = 0.0;
    double min;
    double max;
    int num_contrib = 0;
    for (int i=0; i<pin.size(); i++) {
      PacketPtr curr = pin[i];
      double in_sum;
      double in_sos;
      double in_min;
      double in_max;
      int numData;

      curr->unpack(ToM_Stats_Filter_format_string,
		   &in_sum, &in_sos, &in_min, &in_max, &numData);
      
      if (i == 0) {
	min = in_min;
	max = in_max;
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

    PacketPtr new_packet (new Packet(pin[0]->get_StreamId(),
				     pin[0]->get_Tag(),
				     ToM_Stats_Filter_format_string, 
				     sum, sumOfSquares, min, max,
				     num_contrib));
    pout.push_back(new_packet);
  }

}
