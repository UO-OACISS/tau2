#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdio.h>

// This auditor supports all API versions.
unsigned int la_version(unsigned int version) {
	return version;
}


void la_preinit(uintptr_t *cookie) {
	typedef void (*Tau_init_dl_initialized_p) ();
  static Tau_init_dl_initialized_p Tau_init_dl_initialized_h = NULL;

	void *tau_so = dlmopen(LM_ID_BASE, "libTAU.so", RTLD_NOW);
	if (tau_so == NULL) {
		printf("TAU: ERROR in opening TAU library in auditor.\n");
	}
	else {
		Tau_init_dl_initialized_h = (Tau_init_dl_initialized_p) dlsym(tau_so, "Tau_init_dl_initialized");
		if (Tau_init_dl_initialized_h == NULL) {
			printf("TAU: ERROR obtaining symbol info in auditor.\n");
		}
		else {
			(*Tau_init_dl_initialized_h)();
		}
		dlclose(tau_so);
	}
}
