

void __attribute__ ((constructor)) taupreload_init(void);
void __attribute__ ((destructor)) taupreload_fini(void);

#include <TAU.h>
#include <stdlib.h>

extern void Tau_init_initializeTAU(void);
extern void Tau_profile_exit_all_threads(void);

// Bookend timer covering the lifetime of the preloaded process. Mirrors
// the `taupreload_main` timer the Linux/glibc path creates by interposing
// __libc_start_main; on platforms without that hook (macOS, ppc64) the
// constructor/destructor pair is the next-best approximation. Without
// this, programs that contain no user-instrumented code (e.g. an empty
// Fortran `program`) produce a profile with only the framework's
// `.TAU application` entry under TAU_UTILITY -- and any consumer that
// expects at least one TAU_DEFAULT entry fails.
static void * taupreload_main_handle = NULL;

void taupreload_init() {
  Tau_init_initializeTAU();
  Tau_create_top_level_timer_if_necessary();
  int tmp = TAU_PROFILE_GET_NODE();
  if (tmp == -1) {
    TAU_PROFILE_SET_NODE(0);
  }
  TAU_PROFILER_CREATE(taupreload_main_handle, "taupreload_main", "",
                      TAU_DEFAULT);
  TAU_PROFILER_START(taupreload_main_handle);
}

void taupreload_fini() {
  if (taupreload_main_handle) {
    TAU_PROFILER_STOP(taupreload_main_handle);
  }
  Tau_profile_exit_all_threads();
  Tau_destructor_trigger();
}

