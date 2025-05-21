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


int GPTLsetoption (const int option, const int val) {
    TAU_GPTL_UNIMPLEMENTED();
    (void)option;
    (void)val;
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
    // TODO Create map from int to TAU timer handle
    // For now just start and stop based on name
    *result = 0;
    return 0;
}

int GPTLstart_handle (const char * name, int * handle) {
    // TODO Save timer to avoid lookup by name
    (void)handle;
    TAU_START(name);
    return 0;
}

int GPTLstop (const char * name) {
    TAU_STOP(name);
    return 0;
}

int GPTLstop_handle (const char * name, int * handle) {
    (void)handle;
    TAU_STOP(name);
    return 0;
}

int GPTLstamp (double *wall, double *usr, double *sys) {
    TAU_GPTL_UNIMPLEMENTED();
    (void)wall;
    (void)usr;
    (void)sys;
    return 0;
}

int GPTLpr (const int id) {
    TAU_GPTL_UNIMPLEMENTED();
    (void)id;
    return 0;
}

int GPTLpr_file (const char *outfile) {
    TAU_GPTL_UNIMPLEMENTED();
    (void)outfile;
    return 0;
}
    
int GPTLreset (void) {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;
}

int GPTLreset_timer (const char * name) {
    TAU_GPTL_UNIMPLEMENTED();
    (void)name;
    return 0;
}

int GPTLfinalize (void) {
    Tau_profile_exit_all_threads();
    Tau_destructor_trigger();
    return 0;
}

int GPTLget_memusage (float *usage) {
    TAU_GPTL_UNIMPLEMENTED();
    if(usage != NULL) {
        *usage = 0.0f;
    }
    return 0;

}

int GPTLprint_memusage (const char * str) {
    TAU_GPTL_UNIMPLEMENTED();
    (void)str;
    return 0;

}

int GPTLprint_rusage (const char * str) {
    TAU_GPTL_UNIMPLEMENTED();
    (void)str;
    return 0;

}                                      

int GPTLget_procsiz (float * procsiz_out, float * rss_out) {
    TAU_GPTL_UNIMPLEMENTED();
    if(procsiz_out != NULL) {
        *procsiz_out = 0.0f;
    }
    if(rss_out != NULL) {
        *rss_out = 0.0f;
    }
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

int GPTLsetutr (const int option) {
    TAU_GPTL_UNIMPLEMENTED();
    (void)option;
    return 0;

} 

int GPTLquery (const char *name, int t, int *count, int *onflg, double *wallclock,
        double *dusr, double *dsys, long long *papicounters_out, const int maxcounters) {
    TAU_GPTL_UNIMPLEMENTED();
    (void)name;
    if(onflg != NULL) {
        *onflg = 0;
    }
    if(count != NULL) {
        *count = 0;
    }
    if(wallclock !=  NULL) {
        *wallclock = 0.0;
    }
    if(dusr != NULL) {
        *dusr = 0.0;
    }
    if(dsys != NULL) {
        *dsys = 0.0;
    }
    if(papicounters_out != NULL) {
        *papicounters_out = 0;
    }
    return 0;

} 

int GPTLget_wallclock (const char *timername, int t, double *value)  {
    TAU_GPTL_UNIMPLEMENTED();
    (void)timername;
    (void)t;
    if(value != NULL) {
        *value = 0.0;
    }
    return 0;

} 

int GPTLget_wallclock_latest (const char * timername, int t, double *value) {
    TAU_GPTL_UNIMPLEMENTED();
    (void)timername;
    (void)t;
    if(value != NULL) {
        *value = 0.0;
    }
    return 0;

} 

int GPTLget_threadwork (const char *name, double *maxwork, double *imbal)   {
    TAU_GPTL_UNIMPLEMENTED();
    (void)name;
    if(maxwork != NULL) {
        *maxwork = 0.0;
    }
    if(imbal != NULL) {
        *imbal = 0.0;
    }
    return 0;

} 

int GPTLstartstop_val (const char *name, double value)   {
    TAU_GPTL_UNIMPLEMENTED();
    (void)name;
    (void)value;
    return 0;

} 

int GPTLget_nregions (int t, int * nregions)  {
    TAU_GPTL_UNIMPLEMENTED();
    (void)t;
    if(nregions != NULL) {
        *nregions = 0;
    }
    return 0;

} 

int GPTLget_regionname (int t, int region, char * name, int nc)   {
    TAU_GPTL_UNIMPLEMENTED();
    (void)t;
    (void)region;
    (void)nc;
    if(name != NULL) {
        *name = '\0';
    }
    return 0;

} 

int GPTL_PAPIlibraryinit (void)   {
    TAU_GPTL_UNIMPLEMENTED();
    return 0;

} 

int GPTLevent_name_to_code (const char *name, int *code)   {
    TAU_GPTL_UNIMPLEMENTED();
    (void)name;
    if(code != NULL) {
        *code = 0;
    }
    return 0;

} 

int GPTLevent_code_to_name (const int code, char * name)   {
    TAU_GPTL_UNIMPLEMENTED();
    (void)code;
    if(name != NULL) {
        *name = '\0';
    }
    return 0;

} 

int GPTLget_eventvalue (const char *timername, const char *eventname, int t, double *value)   {
    TAU_GPTL_UNIMPLEMENTED();
    (void)timername;
    (void)eventname;
    (void)t;
    if(value != NULL) {
        *value = 0.0;
    }
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

int GPTLget_count (const char *timername, int t, int *count)   {
    TAU_GPTL_UNIMPLEMENTED();
    (void)timername;
    (void)t;
    if(count != NULL) {
        *count = 0;
    }
    return 0;
} 
