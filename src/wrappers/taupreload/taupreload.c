

void __attribute__ ((constructor)) taupreload_init(void);
void __attribute__ ((destructor)) taupreload_fini(void);

#include <TAU.h>
#include <stdlib.h>

extern void Tau_init_initializeTAU(void);
extern void Tau_profile_exit_all_threads(void);

void taupreload_init() {
  Tau_init_initializeTAU();
  Tau_create_top_level_timer_if_necessary();
  TAU_PROFILE_SET_NODE(0);
}

void taupreload_fini() {
  Tau_profile_exit_all_threads();
  Tau_destructor_trigger();
}
