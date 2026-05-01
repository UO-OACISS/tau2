#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <stdio.h>
#include <TAU.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <signal.h>
#include <Profile/TauEnv.h>
#include <level_zero/ze_api.h>


extern int Tau_init_check_initialized();

typedef ze_result_t (*real_zeInit_t)(ze_init_flags_t);

ZE_APIEXPORT ze_result_t ZE_APICALL zeInit(ze_init_flags_t flags)
{
    static real_zeInit_t real_zeInit = NULL;

    if (real_zeInit == NULL) {
        real_zeInit = (real_zeInit_t)dlsym(RTLD_NEXT, "zeInit");
        if (!real_zeInit) {
            fprintf(stderr, "Failed to find real zeInit\n");
            return ZE_RESULT_ERROR_UNKNOWN;
        }
    }

    if(!Tau_init_check_initialized())
        Tau_init_initializeTAU();

    return real_zeInit(flags);
}


