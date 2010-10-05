#include "Profile/TauMonMrnet.h"

#include "mrnet/Packet.h"
#include "mrnet/NetworkTopology.h"

using namespace MRN;

extern "C" {

  const char *ToM_Cluster_Filter_format_string = "%alf %ad";

  void ToM_Cluster_Filter(std::vector<PacketPtr>& pin,
			  std::vector<PacketPtr>& pout,
			  std::vector<PacketPtr>& /* packets_out_reverse */,
			  void ** /* client data */,
			  PacketPtr& params,
			  const TopologyLocalInfo&) {
    int numCentroidItems;
    int numMemberItems;
    double *changeVector;
    int *changeMembers;
    for (int i=0; i<pin.size(); i++) {
      PacketPtr curr = pin[i];

      int in_numCentroidItems;
      int in_numMemberItems;
      double *in_changeVector;
      int *in_changeMembers;
      if (i == 0) {
	curr->unpack(ToM_Cluster_Filter_format_string,
		     &changeVector, &numCentroidItems,
		     &changeMembers, &numMemberItems);
      } else {
	curr->unpack(ToM_Cluster_Filter_format_string,
		     &in_changeVector, &in_numCentroidItems,
		     &in_changeMembers, &in_numMemberItems);
	// sanity check
	assert((in_numCentroidItems == numCentroidItems) &&
	       (in_numMemberItems == numMemberItems));
	for (int i=0; i<in_numCentroidItems; i++) {
	  changeVector[i] += in_changeVector[i];
	}
	for (int i=0; i<in_numMemberItems; i++) {
	  changeMembers[i] += in_changeMembers[i];
	}
      }
    }

    PacketPtr new_packet (new Packet(pin[0]->get_StreamId(),
				     pin[0]->get_Tag(),
				     ToM_Cluster_Filter_format_string, 
				     changeVector, numCentroidItems,
				     changeMembers, numMemberItems));
    pout.push_back(new_packet);
  }

}
