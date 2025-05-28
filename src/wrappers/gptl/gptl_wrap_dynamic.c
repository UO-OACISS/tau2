#include <Profile/Profiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include <dlfcn.h>

#include "gptl.h"

static const char * gptl_orig_libname = "libgptl.so";
static void * gptl_handle = NULL;

static bool gptl_enabled = true;

extern void Tau_profile_exit_all_threads(void);
extern int Tau_init_initializeTAU(void);
extern void Tau_pure_increment(const char * n, double time, int calls);

#define TAU_GPTL_UNIMPLEMENTED() \
    do { \
        static bool first = true; \
        if(first) { \
            fprintf(stderr, "TAU: Warning: GPTL function %s is not implemented by TAU.\n", __func__); \
            first = false; \
        } \
    } while(0)



int GPTLinitialize (void) {
    int result = 0;

    Tau_init_initializeTAU();
    Tau_create_top_level_timer_if_necessary();
    gptl_enabled = true;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
        if(gptl_handle == NULL) {
            fprintf(stderr, "TAU: Warning: could not load GPTL library %s. Is it in your LD_LIBRARY_PATH?\n%s\n", gptl_orig_libname, dlerror());
        }
    }

    if(gptl_handle != NULL) {
        static int (*GPTLinitialize_h)(void) = NULL;
        if(GPTLinitialize_h == NULL) {
            GPTLinitialize_h = dlsym(gptl_handle, "GPTLinitialize");
        }
        if(GPTLinitialize_h != NULL) {
            result = (*GPTLinitialize_h)();
        }
    }

    return result;
}

int GPTLsetoption (const int option, const int val) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLsetoption_h)(const int option, const int val) = NULL;
        if(GPTLsetoption_h == NULL) {
            GPTLsetoption_h = dlsym(gptl_handle, "GPTLsetoption");
        }
        if(GPTLsetoption_h != NULL) {
            result = (*GPTLsetoption_h)(option, val);
        }
    }

    return result;
}


int GPTLstart (const char * name) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLstart_h)(const char * name) = NULL;
        if(GPTLstart_h == NULL) {
            GPTLstart_h = dlsym(gptl_handle, "GPTLstart");
        }
        if(GPTLstart_h != NULL) {
            result = (*GPTLstart_h)(name);
        }
    }

    if(gptl_enabled) {
        TAU_START(name);
    }
    return result;
}

int GPTLinit_handle (const char * name, int * result) {
    int status = 0;
    *result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLinit_handle_h)(const char * name, int * result) = NULL;
        if(GPTLinit_handle_h == NULL) {
            GPTLinit_handle_h = dlsym(gptl_handle, "GPTLinit_handle");
        }
        if(GPTLinit_handle_h != NULL) {
            status = (*GPTLinit_handle_h)(name, result);
        }
    }

    return status;
}

int GPTLstart_handle (const char * name, int * handle) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLstart_handle_h)(const char * name, int * handle) = NULL;
        if(GPTLstart_handle_h == NULL) {
            GPTLstart_handle_h = dlsym(gptl_handle, "GPTLstart_handle");
        }
        if(GPTLstart_handle_h != NULL) {
            result = (*GPTLstart_handle_h)(name, handle);
        }
    }

    if(gptl_enabled) {
        TAU_START(name);
    }
    return result;
}

int GPTLstop (const char * name) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLstop_h)(const char * name) = NULL;
        if(GPTLstop_h == NULL) {
            GPTLstop_h = dlsym(gptl_handle, "GPTLstop");
        }
        if(GPTLstop_h != NULL) {
            result = (*GPTLstop_h)(name);
        }
    }

    if(gptl_enabled) {
        TAU_STOP(name);
    }
    return result;
}

int GPTLstop_handle (const char * name, int * handle) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLstop_handle_h)(const char * name, int * handle) = NULL;
        if(GPTLstop_handle_h == NULL) {
            GPTLstop_handle_h = dlsym(gptl_handle, "GPTLstop_handle");
        }
        if(GPTLstop_handle_h != NULL) {
            result = (*GPTLstop_handle_h)(name, handle);
        }
    }

    if(gptl_enabled) {
        TAU_STOP(name);
    }
    return result;
}

int GPTLstamp (double *wall, double *usr, double *sys) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLstamp_h)(double * wall, double * usr, double * sys) = NULL;
        if(GPTLstamp_h == NULL) {
            GPTLstamp_h = dlsym(gptl_handle, "GPTLstamp");
        }
        if(GPTLstamp_h != NULL) {
            result = (*GPTLstamp_h)(wall, usr, sys);
        }
    }

    return result;
}

int GPTLpr (const int id) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLpr_h)(const int id) = NULL;
        if(GPTLpr_h == NULL) {
            GPTLpr_h = dlsym(gptl_handle, "GPTLpr");
        }
        if(GPTLpr_h != NULL) {
            result = (*GPTLpr_h)(id);
        }
    }

    return result;
}

int GPTLpr_file (const char *outfile) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLpr_file_h)(const char * outfile) = NULL;
        if(GPTLpr_file_h == NULL) {
            GPTLpr_file_h = dlsym(gptl_handle, "GPTLpr_file");
        }
        if(GPTLpr_file_h != NULL) {
            result = (*GPTLpr_file_h)(outfile);
        }
    }

    return result;
}
    
int GPTLreset (void) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLreset_h)(void) = NULL;
        if(GPTLreset_h == NULL) {
            GPTLreset_h = dlsym(gptl_handle, "GPTLreset");
        }
        if(GPTLreset_h != NULL) {
            result = (*GPTLreset_h)();
        }
    }

    TAU_GPTL_UNIMPLEMENTED();
    return result;
}

int GPTLreset_timer (const char * name) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLreset_timer_h)(const char * name) = NULL;
        if(GPTLreset_timer_h == NULL) {
            GPTLreset_timer_h = dlsym(gptl_handle, "GPTLreset_timer");
        }
        if(GPTLreset_timer_h != NULL) {
            result = (*GPTLreset_timer_h)(name);
        }
    }

    TAU_GPTL_UNIMPLEMENTED();
    return result;
}

int GPTLfinalize (void) {
    int result = 0;
    Tau_profile_exit_all_threads();
    Tau_destructor_trigger();

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLfinalize_h)(void) = NULL;
        if(GPTLfinalize_h == NULL) {
            GPTLfinalize_h = dlsym(gptl_handle, "GPTLfinalize");
        }
        if(GPTLfinalize_h != NULL) {
            result = (*GPTLfinalize_h)();
        }
    }

    return result;
}

int GPTLget_memusage (float *usage) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLget_memusage_h)(float * usage) = NULL;
        if(GPTLget_memusage_h == NULL) {
            GPTLget_memusage_h = dlsym(gptl_handle, "GPTLget_memusage");
        }
        if(GPTLget_memusage_h != NULL) {
            result = (*GPTLget_memusage_h)(usage);
        }
    }

    return result;
}

int GPTLprint_memusage (const char * str) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLprint_memusage_h)(const char * str) = NULL;
        if(GPTLprint_memusage_h == NULL) {
            GPTLprint_memusage_h = dlsym(gptl_handle, "GPTLprint_memusage");
        }
        if(GPTLprint_memusage_h != NULL) {
            result = (*GPTLprint_memusage_h)(str);
        }
    }

    return result;
}

int GPTLprint_rusage (const char * str) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLprint_rusage_h)(const char * str) = NULL;
        if(GPTLprint_rusage_h == NULL) {
            GPTLprint_rusage_h = dlsym(gptl_handle, "GPTLprint_rusage");
        }
        if(GPTLprint_rusage_h != NULL) {
            result = (*GPTLprint_rusage_h)(str);
        }
    }

    return result;
}                                      

int GPTLget_procsiz (float * procsiz_out, float * rss_out) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLget_procsiz_h)(float * procsiz_out, float * rss_out) = NULL;
        if(GPTLget_procsiz_h == NULL) {
            GPTLget_procsiz_h = dlsym(gptl_handle, "GPTLget_procsiz");
        }
        if(GPTLget_procsiz_h != NULL) {
            result = (*GPTLget_procsiz_h)(procsiz_out, rss_out);
        }
    }

    return result;
} 

int GPTLenable (void) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLenable_h)(void) = NULL;
        if(GPTLenable_h == NULL) {
            GPTLenable_h = dlsym(gptl_handle, "GPTLenable");
        }
        if(GPTLenable_h != NULL) {
            result = (*GPTLenable_h)();
        }
    }

    gptl_enabled = true;
    return result;
} 

int GPTLdisable (void) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLdisable_h)(void) = NULL;
        if(GPTLdisable_h == NULL) {
            GPTLdisable_h = dlsym(gptl_handle, "GPTLdisable");
        }
        if(GPTLdisable_h != NULL) {
            result = (*GPTLdisable_h)();
        }
    }

    gptl_enabled = false;
    return result;
} 

int GPTLsetutr (const int option) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLsetutr_h)(const int option) = NULL;
        if(GPTLsetutr_h == NULL) {
            GPTLsetutr_h = dlsym(gptl_handle, "GPTLsetutr");
        }
        if(GPTLsetutr_h != NULL) {
            result = (*GPTLsetutr_h)(option);
        }
    }

    return result;
} 

int GPTLquery (const char *name, int t, int *count, int *onflg, double *wallclock,
        double *dusr, double *dsys, long long *papicounters_out, const int maxcounters) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLquery_h)(const char *name, int t, int *count, int *onflg, double *wallclock,
            double *dusr, double *dsys, long long *papicounters_out, const int maxcounters) = NULL;
        if(GPTLquery_h == NULL) {
            GPTLquery_h = dlsym(gptl_handle, "GPTLquery");
        }
        if(GPTLquery_h != NULL) {
            result = (*GPTLquery_h)(name, t, count, onflg, wallclock, dusr, dsys, papicounters_out, maxcounters);
        }
    }

    return result;
} 

int GPTLget_wallclock (const char *timername, int t, double *value)  {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLget_wallclock_h)(const char * timername, int t, double * value) = NULL;
        if(GPTLget_wallclock_h == NULL) {
            GPTLget_wallclock_h = dlsym(gptl_handle, "GPTLget_wallclock");
        }
        if(GPTLget_wallclock_h != NULL) {
            result = (*GPTLget_wallclock_h)(timername, t, value);
        }
    }

    return result;
} 

int GPTLget_wallclock_latest (const char * timername, int t, double *value) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLget_wallclock_latest_h)(const char * timername, int t, double * value) = NULL;
        if(GPTLget_wallclock_latest_h == NULL) {
            GPTLget_wallclock_latest_h = dlsym(gptl_handle, "GPTLget_wallclock_latest");
        }
        if(GPTLget_wallclock_latest_h != NULL) {
            result = (*GPTLget_wallclock_latest_h)(timername, t, value);
        }
    }

    return result;
} 

int GPTLget_threadwork (const char *name, double *maxwork, double *imbal)   {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLget_threadwork_h)(const char * name, double * maxwork, double * imbal) = NULL;
        if(GPTLget_threadwork_h == NULL) {
            GPTLget_threadwork_h = dlsym(gptl_handle, "GPTLget_threadwork");
        }
        if(GPTLget_threadwork_h != NULL) {
            result = (*GPTLget_threadwork_h)(name, maxwork, imbal);
        }
    }

    return result;
} 

int GPTLstartstop_val (const char *name, double value)   {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLstartstop_val_h)(const char * name, double value) = NULL;
        if(GPTLstartstop_val_h == NULL) {
            GPTLstartstop_val_h = dlsym(gptl_handle, "GPTLstartstop_val");
        }
        if(GPTLstartstop_val_h != NULL) {
            result = (*GPTLstartstop_val_h)(name, value);
        }
    }

    // TODO is unit conversion needed?
    Tau_pure_increment(name, value, 1);
    return result;
} 

int GPTLget_nregions (int t, int * nregions)  {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLget_nregions_h)(int t, int * nregions) = NULL;
        if(GPTLget_nregions_h == NULL) {
            GPTLget_nregions_h = dlsym(gptl_handle, "GPTLget_nregions");
        }
        if(GPTLget_nregions_h != NULL) {
            result = (*GPTLget_nregions_h)(t, nregions);
        }
    }

    return result;
} 

int GPTLget_regionname (int t, int region, char * name, int nc)   {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLget_regionname_h)(int t, int region, char * name, int nc) = NULL;
        if(GPTLget_regionname_h == NULL) {
            GPTLget_regionname_h = dlsym(gptl_handle, "GPTLget_regionname");
        }
        if(GPTLget_regionname_h != NULL) {
            result = (*GPTLget_regionname_h)(t, region, name, nc);
        }
    }

    return result;
} 

int GPTL_PAPIlibraryinit (void)   {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTL_PAPIlibraryinit_h)(void) = NULL;
        if(GPTL_PAPIlibraryinit_h == NULL) {
            GPTL_PAPIlibraryinit_h = dlsym(gptl_handle, "GPTL_PAPIlibraryinit");
        }
        if(GPTL_PAPIlibraryinit_h != NULL) {
            result = (*GPTL_PAPIlibraryinit_h)();
        }
    }

    return result;
} 

int GPTLevent_name_to_code (const char *name, int *code)   {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLevent_name_to_code_h)(const char * name, int * code) = NULL;
        if(GPTLevent_name_to_code_h == NULL) {
            GPTLevent_name_to_code_h = dlsym(gptl_handle, "GPTLevent_name_to_code");
        }
        if(GPTLevent_name_to_code_h != NULL) {
            result = (*GPTLevent_name_to_code_h)(name, code);
        }
    }

    return result;
} 

int GPTLevent_code_to_name (const int code, char * name)   {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLevent_code_to_name_h)(const int code, char * name) = NULL;
        if(GPTLevent_code_to_name_h == NULL) {
            GPTLevent_code_to_name_h = dlsym(gptl_handle, "GPTLevent_code_to_name");
        }
        if(GPTLevent_code_to_name_h != NULL) {
            result = (*GPTLevent_code_to_name_h)(code, name);
        }
    }

    return result;
} 

int GPTLget_eventvalue (const char *timername, const char *eventname, int t, double *value)   {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLget_eventvalue_h)(const char * timername, const char * eventname, double * value) = NULL;
        if(GPTLget_eventvalue_h == NULL) {
            GPTLget_eventvalue_h = dlsym(gptl_handle, "GPTLget_eventvalue");
        }
        if(GPTLget_eventvalue_h != NULL) {
            result = (*GPTLget_eventvalue_h)(timername, eventname, value);
        }
    }

    return result;
} 

int GPTLnum_errors (void) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLnum_errors_h)(void) = NULL;
        if(GPTLnum_errors_h == NULL) {
            GPTLnum_errors_h = dlsym(gptl_handle, "GPTLnum_errors");
        }
        if(GPTLnum_errors_h != NULL) {
            result = (*GPTLnum_errors_h)();
        }
    }

    return result;
} 

int GPTLnum_warn (void) {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLnum_warn_h)(void) = NULL;
        if(GPTLnum_warn_h == NULL) {
            GPTLnum_warn_h = dlsym(gptl_handle, "GPTLnum_warn");
        }
        if(GPTLnum_warn_h != NULL) {
            result = (*GPTLnum_warn_h)();
        }
    }

    return result;
} 

int GPTLget_count (const char *timername, int t, int *count)   {
    int result = 0;

    if(gptl_handle == NULL) {
        gptl_handle = (void *)dlopen(gptl_orig_libname, RTLD_NOW);
    }

    if(gptl_handle != NULL) {
        static int (*GPTLget_count_h)(const char * timername, int t, int * count) = NULL;
        if(GPTLget_count_h == NULL) {
            GPTLget_count_h = dlsym(gptl_handle, "GPTLget_count");
        }
        if(GPTLget_count_h != NULL) {
            result = (*GPTLget_count_h)(timername, t, count);
        }
    }

    return result;
} 

