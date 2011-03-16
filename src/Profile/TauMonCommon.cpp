#include "Profile/TauMon.h"

/* API wrapper implementation. Empty if no monitoring transport is
   provided.
*/
extern "C" void Tau_mon_onlineDump() {
#ifdef TAU_MONITORING
  Tau_mon_internal_onlineDump();
#endif /* TAU_MONITORING */
}
