#ifdef TAU_TRACK_LD_LOADER

#define _GNU_SOURCE
#include <dlfcn.h>
#ifdef MERCURIUM_EXTRA
#define LM_ID_BASE 0
#endif
#include <link.h>
#include <stdio.h>

// This auditor supports all API versions.
unsigned int la_version(unsigned int version)
{
  return version;
}

void la_preinit(uintptr_t *cookie)
{
  typedef void (*Tau_init_dl_initialized_p)();
  static Tau_init_dl_initialized_p Tau_init_dl_initialized_h = NULL;
  void * tau_so;

  tau_so = dlmopen(LM_ID_BASE, "libTAU.so", RTLD_NOW);

  if (tau_so) {
    char const * err;

    dlerror(); // reset error flag
    Tau_init_dl_initialized_h = (Tau_init_dl_initialized_p)dlsym(tau_so, "Tau_init_dl_initialized");
    // Check for errors
    if ((err = dlerror())) {
      printf("TAU: ERROR obtaining symbol info in auditor: %s\n", err);
    } else {
      Tau_init_dl_initialized_h();
    }
    dlclose(tau_so);
  } else {
    printf("TAU: ERROR in opening TAU library in auditor.\n");
  }
}

#endif //TAU_TRACK_LD_LOADER
