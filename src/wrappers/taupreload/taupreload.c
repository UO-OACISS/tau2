

void __attribute__ ((constructor)) taupreload_init(void);
void __attribute__ ((destructor)) taupreload_fini(void);

#include <TAU.h>
#include <stdlib.h>

void taupreload_init() {
  Tau_global_incr_insideTAU();
  Tau_create_top_level_timer_if_necessary();
  Tau_global_decr_insideTAU();

  TAU_PROFILE_SET_NODE(0);
}

void taupreload_fini() {
  Tau_stop_top_level_timer_if_necessary();
}
