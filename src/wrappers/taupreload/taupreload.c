

void __attribute__ ((constructor)) taupreload_init(void);
void __attribute__ ((destructor)) taupreload_fini(void);

#include <TAU.h>
#include <stdlib.h>

void taupreload_init() {
  Tau_init_initializeTAU();
  Tau_create_top_level_timer_if_necessary();
  TAU_PROFILE_SET_NODE(0);
}

void taupreload_fini() {
  Tau_destructor_trigger();
  Tau_profile_exit_all_threads();
}
