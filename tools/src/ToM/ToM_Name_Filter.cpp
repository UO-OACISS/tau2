#include "Profile/TauMonMrnet.h"

#include "mrnet/Packet.h"
#include "mrnet/NetworkTopology.h"

#include <string.h>

using namespace MRN;

extern "C" {

  // This is really a hack filter. Takes only the strings from
  //   Rank 0 since 0 gets the unified ordered-name list.
  const char *ToM_Name_Filter_format_string = 
    "%d %d %as";

  void ToM_Name_Filter(std::vector<PacketPtr>& pin,
			std::vector<PacketPtr>& pout,
			std::vector<PacketPtr>& /* packets_out_reverse */,
			void ** /* client data */,
			PacketPtr& params,
			const TopologyLocalInfo&) {

    int in_rank = -1;
    int in_numMetrics = 0;
    char **in_name_strings;
    int in_num_strings = 0;

    int found = 0;

    for (int i=0; i<pin.size(); i++) {
      PacketPtr curr = pin[i];
      
      curr->unpack(ToM_Name_Filter_format_string, &in_rank, 
		   &in_numMetrics, &in_name_strings, &in_num_strings);

      if (in_rank == 0) {
	// done. break the loop and copy the data for this rank
	//   to the output port. 
	found = 1;
	break;
      }
    }

    if (in_rank != 0) {
      in_num_strings = 1;
      in_numMetrics = 0;
      in_name_strings = (char **)malloc(sizeof(char *));
      in_name_strings[0] = (char *)malloc(sizeof(char));
      strcpy(in_name_strings[0],"");
    }
    PacketPtr new_packet (new Packet(pin[0]->get_StreamId(),
				     pin[0]->get_Tag(),
				     ToM_Name_Filter_format_string, 
				     in_rank, in_numMetrics, 
				     in_name_strings,
				     in_num_strings));

    pout.push_back(new_packet);
  }

}
