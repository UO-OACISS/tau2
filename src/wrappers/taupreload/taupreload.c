

void __attribute__ ((constructor)) taupreload_init(void);
void __attribute__ ((destructor)) taupreload_fini(void);

#include <TAU.h>
#include <stdlib.h>

void taupreload_init() {
  Tau_create_top_level_timer_if_necessary();
  if (Tau_get_tid() == 0) {
    Tau_global_incr_insideTAU();
    char *c = calloc(1,8); /* TAU calls calloc: memory wrapper */
    Tau_global_decr_insideTAU();
    free(c);
  }
  
  TAU_PROFILE_SET_NODE(0);
  //  TAU_START(".TAU Application!");
}

void taupreload_fini() {
  Tau_stop_top_level_timer_if_necessary();
  //TAU_STOP(".TAU Application!");
}
