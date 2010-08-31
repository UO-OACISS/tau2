#include "Profile/TauMonMrnet.h"

#include "mrnet/Packet.h"
#include "mrnet/NetworkTopology.h"

using namespace MRN;

extern "C" {

  const char *ToM_Histogram_Filter_format_string = "%ad";

  void ToM_Histogram_Filter(std::vector<PacketPtr>& pin,
			    std::vector<PacketPtr>& pout,
			    std::vector<PacketPtr>& /* packets_out_reverse */,
			    void ** /* client data */,
			    PacketPtr& params,
			    const TopologyLocalInfo&) {
    int numDataItems;
    int *histBins;

    for (int i=0; i<pin.size(); i++) {
      PacketPtr curr = pin[i];

      int in_numDataItems;
      int *in_histBins;
      if (i == 0) {
	curr->unpack(ToM_Histogram_Filter_format_string,
		     &histBins, &numDataItems);
      } else {
	curr->unpack(ToM_Histogram_Filter_format_string,
		     &in_histBins, &in_numDataItems);
	if (in_numDataItems != numDataItems) {
	  printf("Histogram Filter ERROR: Incorrectly sized bins!\n");
	}
	for (int i=0; i<in_numDataItems; i++) {
	  histBins[i] += in_histBins[i];
	}
      }
    }

    PacketPtr new_packet (new Packet(pin[0]->get_StreamId(),
				     pin[0]->get_Tag(),
				     ToM_Histogram_Filter_format_string, 
				     histBins, numDataItems));
    pout.push_back(new_packet);
  }

}
