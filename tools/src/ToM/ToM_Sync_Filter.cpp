#include "Profile/TauMonMrnet.h"

#include "mrnet/Packet.h"
#include "mrnet/NetworkTopology.h"

using namespace MRN;

// This is only being implemented to allow the FE to wait for a single 
//   combined message from all the back-ends. Apparently, TFILTER_NULL
//   will simply let each individual packet flow up the tree even if
//   SFILTER_WAITFORALL (default case) is active.
//
// Our implementation will simply copy the data from the first child,
//   but forward it only when all children have reported in via
//   SFILTER_WAITFORALL.
extern "C" {

  const char *ToM_Sync_Filter_format_string = "%d";

  void ToM_Sync_Filter(std::vector<PacketPtr>& pin,
		       std::vector<PacketPtr>& pout,
		       std::vector<PacketPtr>& /* packets_out_reverse */,
		       void ** /* client data */,
		       PacketPtr& params,
		       const TopologyLocalInfo&) {

    PacketPtr curr = pin[0];
    int data;

    curr->unpack(ToM_Sync_Filter_format_string, &data);
		 
    PacketPtr new_packet (new Packet(pin[0]->get_StreamId(),
				     pin[0]->get_Tag(),
				     ToM_Sync_Filter_format_string, 
				     data));
    pout.push_back(new_packet);
  }

}
