#ifdef TAU_TRACK_LD_LOADER

#define _GNU_SOURCE
#include <dlfcn.h>
#ifdef MERCURIUM_EXTRA
#define LM_ID_BASE 0
#endif
#include <link.h>
#include <stdio.h>


int * objopen_counter()
{
  static int count = 0;
  return &count;
}

// This auditor supports all API versions.
unsigned int la_version(unsigned int version)
{
  return version;
}

#if 1
unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie)
{
  (*objopen_counter())++;
  return 0;
}

void la_preinit(uintptr_t *cookie)
{
  typedef void (*Tau_init_dl_initialized_t)();
  typedef void (*Tau_bfd_register_objopen_counter_t)(int * (*)(void));
  static Tau_init_dl_initialized_t Tau_init_dl_initialized = NULL;
  static Tau_bfd_register_objopen_counter_t Tau_bfd_register_objopen_counter = NULL;
  void * tau_so;

  tau_so = dlmopen(LM_ID_BASE, "libTAU.so", RTLD_NOW);

  if (tau_so) {
    char const * err;

    dlerror(); // reset error flag
    Tau_init_dl_initialized = (Tau_init_dl_initialized_t)dlsym(tau_so, "Tau_init_dl_initialized");
    Tau_bfd_register_objopen_counter = (Tau_bfd_register_objopen_counter_t)dlsym(tau_so, "Tau_bfd_register_objopen_counter");
    // Check for errors
    if ((err = dlerror())) {
      printf("TAU: ERROR obtaining symbol info in auditor: %s\n", err);
    } else {
      Tau_init_dl_initialized();
      Tau_bfd_register_objopen_counter(objopen_counter);
    }
    dlclose(tau_so);
  } else {
    printf("TAU: ERROR in opening TAU library in auditor.\n");
  }
}
#endif

#endif //TAU_TRACK_LD_LOADER
