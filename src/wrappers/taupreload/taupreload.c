

void __attribute__ ((constructor)) taupreload_init(void);
void __attribute__ ((destructor)) taupreload_fini(void);

#include <TAU.h>

void taupreload_init() {
  Tau_create_top_level_timer_if_necessary();
  TAU_PROFILE_SET_NODE(0);
  //  TAU_START(".TAU Application!");
}

void taupreload_fini() {
  Tau_stop_top_level_timer_if_necessary();
  //TAU_STOP(".TAU Application!");
}
