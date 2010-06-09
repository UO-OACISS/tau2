#include "Profile/TauMonMrnet.h"

#include "mrnet/Packet.h"
#include "mrnet/NetworkTopology.h"

using namespace MRN;

extern "C" {

  const char *ToM_Test_Filter_format_string = "%f %d";

  void ToM_Test_Filter(std::vector<PacketPtr>& pin,
		       std::vector<PacketPtr>& pout,
		       std::vector<PacketPtr>& /* packets_out_reverse */,
		       void ** /* client data */,
		       PacketPtr& params,
		       const TopologyLocalInfo&) {
    double total;
    int num_contrib = 0;
    for (int i=0; i<pin.size(); i++) {
      PacketPtr curr = pin[i];
      double data;
      int numData;

      curr->unpack(ToM_Test_Filter_format_string,
		   &data, &numData);
      total += data;
      num_contrib += numData;
    }

    total /= num_contrib;

    PacketPtr new_packet (new Packet(pin[0]->get_StreamId(),
				     pin[0]->get_Tag(),
				     ToM_Test_Filter_format_string, 
				     total, num_contrib));
    pout.push_back(new_packet);
  }

}
