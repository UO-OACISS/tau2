#include <Profile/Profiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include "gptl.h"

extern void Tau_profile_exit_all_threads(void);
extern int Tau_init_initializeTAU(void);

#define TAU_GPTL_UNIMPLEMENTED() \
    do { \
        static bool first = true; \
        if(first) { \
            fprintf(stderr, "TAU: Warning: GPTL function %s is not implemented by TAU.\n", __func__); \
            first = false; \
        } \
    } while(0)


int GPTLsetoption (const int, const int) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;
}

int GPTLinitialize (void) {
    int result = Tau_init_initializeTAU();
    Tau_create_top_level_timer_if_necessary();
    return result;
}

int GPTLstart (const char * name) {
    TAU_START(name);
    return 0;
}

int GPTLinit_handle (const char * name, int * result) {
    TAU_PROFILE_TIMER(timer, name, "", TAU_DEFAULT); 
    result = (int *)timer;
    return 0;
}

int GPTLstart_handle (const char * name, int * handle) {
    (void)name;
    void * timer = (void *)handle;
    TAU_PROFILE_START(timer);
    return 0;
}

int GPTLstop (const char * name) {
    TAU_STOP(name);
    return 0;
}

int GPTLstop_handle (const char * name, int * handle) {
    (void)name;
    void * timer = (void *)handle;
    TAU_PROFILE_STOP(timer);
    return 0;
}

int GPTLstamp (double *, double *, double *) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;
}

int GPTLpr (const int) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;
}

int GPTLpr_file (const char *) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;
}
    
int GPTLreset (void) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;
}

int GPTLreset_timer (const char *) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;
}

int GPTLfinalize (void) {
    Tau_profile_exit_all_threads();
    Tau_destructor_trigger();
    return 0;
}

int GPTLget_memusage (float *) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

}

int GPTLprint_memusage (const char *) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

}

int GPTLprint_rusage (const char *) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

}                                      

int GPTLget_procsiz (float *, float *) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLenable (void) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLdisable (void) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLsetutr (const int) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLquery (const char *, int, int *, int *, double *, double *, double *,
       long long *, const int) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLget_wallclock (const char *, int, double *)  {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLget_wallclock_latest (const char *, int, double *)   {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLget_threadwork (const char *, double *, double *)   {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLstartstop_val (const char *, double)   {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLget_nregions (int, int *)  {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLget_regionname (int, int, char *, int)   {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTL_PAPIlibraryinit (void)   {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLevent_name_to_code (const char *, int *)   {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLevent_code_to_name (const int, char *)   {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLget_eventvalue (const char *, const char *, int, double *)   {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLnum_errors (void)   {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLnum_warn (void)      {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLget_count (const char *, int, int *)   {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 
