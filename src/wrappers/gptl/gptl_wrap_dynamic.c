#define _GNU_SOURCE

#include <Profile/Profiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#include <dlfcn.h>

#include "gptl.h"

static bool gptl_enabled = true;
static char * gptl_prefix = NULL;

extern void Tau_profile_exit_all_threads(void);
extern int Tau_init_initializeTAU(void);
extern void Tau_pure_increment(const char * n, double time, int calls);

#define TAU_GPTL_DEBUG

#define TAU_GPTL_UNIMPLEMENTED() \
    do { \
        static bool first = true; \
        if(first) { \
            fprintf(stderr, "TAU: Warning: GPTL function %s is not implemented by TAU.\n", __func__); \
            first = false; \
        } \
    } while(0)


#ifdef TAU_GPTL_DEBUG
#define TAU_GPTL_LOG() \
    do { \
        fprintf(stderr, "[TAU GPTL] %s\n", __func__); \
    } while(0)

#define TAU_GPTL_LOG_NAME(name) \
    do { \
        fprintf(stderr, "[TAU GPTL] %s(\"%s\")\n", __func__, name); \
    } while(0)
#else // TAU_GPTL_DEBUG
#define TAU_GPTL_LOG()
#define TAU_GPTL_LOG_NAME(name)
#endif // TAU_GPTL_DEBUG


static char * gptl_prefixed_name(const char * name) {
    TAU_GPTL_LOG_NAME(name);
    char * result;
    if(gptl_prefix != NULL) {
        size_t len = strlen(name) + strlen(gptl_prefix);
        char prefixed_name[len + 1];
        snprintf(result, sizeof(prefixed_name), "%s%s", gptl_prefix, name);
        result = strdup(prefixed_name);
    } else {
        result = strdup(name);
    }
    return result;
}

static void gptl_tau_start(const char * name) {
    if(gptl_prefix != NULL) {
        char * timername = gptl_prefixed_name(name);
        TAU_GPTL_LOG_NAME(timername);
        TAU_START(timername);
        free(timername);
    } else {
        TAU_START(name);
    }
}

static void gptl_tau_stop(const char * name) {
    if(gptl_prefix != NULL) {
        char * timername = gptl_prefixed_name(name);
        TAU_GPTL_LOG_NAME(timername);
        TAU_STOP(timername);
        free(timername);
    } else {
        TAU_STOP(name);
    }
}

static void gptl_tau_increment(const char * name, const double add_time, const int add_count) {
    if(gptl_prefix != NULL) {
        char * timername = gptl_prefixed_name(name);
        TAU_GPTL_LOG_NAME(timername);
        // GPTL uses seconds for time
        Tau_pure_increment(timername, add_time * 1000.0 * 1000.0, add_count);
        free(timername);
    } else {
        Tau_pure_increment(name, add_time * 1000.0 * 1000.0, add_count);
    }
}

int GPTLinitialize (void) {
    TAU_GPTL_LOG();
    int result = 0;

    Tau_init_initializeTAU();
    Tau_create_top_level_timer_if_necessary();
    gptl_enabled = true;

    static int (*GPTLinitialize_h)(void) = NULL;
    if(GPTLinitialize_h == NULL) {
        GPTLinitialize_h = dlsym(RTLD_NEXT, "GPTLinitialize");
    }
    if(GPTLinitialize_h == NULL) {
        fprintf(stderr, "TAU: Warning: could not find real libgptl.so\n");
    }
    if(GPTLinitialize_h != NULL) {
        result = (*GPTLinitialize_h)();
    } 

    return result;
}

int GPTLsetoption (const int option, const int val) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLsetoption_h)(const int option, const int val) = NULL;
    if(GPTLsetoption_h == NULL) {
        GPTLsetoption_h = dlsym(RTLD_NEXT, "GPTLsetoption");
    }
    if(GPTLsetoption_h != NULL) {
        result = (*GPTLsetoption_h)(option, val);
    }

    return result;
}


int GPTLstart (const char * name) {
    TAU_GPTL_LOG_NAME(name);
    int result = 0;

    static int (*GPTLstart_h)(const char * name) = NULL;
    if(GPTLstart_h == NULL) {
        GPTLstart_h = dlsym(RTLD_NEXT, "GPTLstart");
    }
    if(GPTLstart_h != NULL) {
        result = (*GPTLstart_h)(name);
    }

    if(gptl_enabled) {
        gptl_tau_start(name);
    }
    return result;
}

int GPTLinit_handle (const char * name, int * result) {
    TAU_GPTL_LOG_NAME(name);
    int status = 0;
    *result = 0;

    static int (*GPTLinit_handle_h)(const char * name, int * result) = NULL;
    if(GPTLinit_handle_h == NULL) {
        GPTLinit_handle_h = dlsym(RTLD_NEXT, "GPTLinit_handle");
    }
    if(GPTLinit_handle_h != NULL) {
        status = (*GPTLinit_handle_h)(name, result);
    }

    return status;
}

int GPTLstart_handle (const char * name, int * handle) {
    TAU_GPTL_LOG_NAME(name);
    int result = 0;

    static int (*GPTLstart_handle_h)(const char * name, int * handle) = NULL;
    if(GPTLstart_handle_h == NULL) {
        GPTLstart_handle_h = dlsym(RTLD_NEXT, "GPTLstart_handle");
    }
    if(GPTLstart_handle_h != NULL) {
        result = (*GPTLstart_handle_h)(name, handle);
    }

    if(gptl_enabled) {
        gptl_tau_start(name);
    }
    return result;
}

int GPTLstop (const char * name) {
    TAU_GPTL_LOG_NAME(name);
    int result = 0;

    static int (*GPTLstop_h)(const char * name) = NULL;
    if(GPTLstop_h == NULL) {
        GPTLstop_h = dlsym(RTLD_NEXT, "GPTLstop");
    }
    if(GPTLstop_h != NULL) {
        result = (*GPTLstop_h)(name);
    }

    if(gptl_enabled) {
        gptl_tau_stop(name);
    }
    return result;
}

int GPTLstop_handle (const char * name, int * handle) {
    TAU_GPTL_LOG_NAME(name);
    int result = 0;

    static int (*GPTLstop_handle_h)(const char * name, int * handle) = NULL;
    if(GPTLstop_handle_h == NULL) {
        GPTLstop_handle_h = dlsym(RTLD_NEXT, "GPTLstop_handle");
    }
    if(GPTLstop_handle_h != NULL) {
        result = (*GPTLstop_handle_h)(name, handle);
    }

    if(gptl_enabled) {
        gptl_tau_stop(name);
    }
    return result;
}

int GPTLstamp (double *wall, double *usr, double *sys) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLstamp_h)(double * wall, double * usr, double * sys) = NULL;
    if(GPTLstamp_h == NULL) {
        GPTLstamp_h = dlsym(RTLD_NEXT, "GPTLstamp");
    }
    if(GPTLstamp_h != NULL) {
        result = (*GPTLstamp_h)(wall, usr, sys);
    }

    return result;
}

int GPTLpr (const int id) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLpr_h)(const int id) = NULL;
    if(GPTLpr_h == NULL) {
        GPTLpr_h = dlsym(RTLD_NEXT, "GPTLpr");
    }
    if(GPTLpr_h != NULL) {
        result = (*GPTLpr_h)(id);
    }

    return result;
}

int GPTLpr_file (const char *outfile) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLpr_file_h)(const char * outfile) = NULL;
    if(GPTLpr_file_h == NULL) {
        GPTLpr_file_h = dlsym(RTLD_NEXT, "GPTLpr_file");
    }
    if(GPTLpr_file_h != NULL) {
        result = (*GPTLpr_file_h)(outfile);
    }

    return result;
}
    
int GPTLreset (void) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLreset_h)(void) = NULL;
    if(GPTLreset_h == NULL) {
        GPTLreset_h = dlsym(RTLD_NEXT, "GPTLreset");
    }
    if(GPTLreset_h != NULL) {
        result = (*GPTLreset_h)();
    }

    TAU_GPTL_UNIMPLEMENTED();
    return result;
}

int GPTLreset_timer (const char * name) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLreset_timer_h)(const char * name) = NULL;
    if(GPTLreset_timer_h == NULL) {
        GPTLreset_timer_h = dlsym(RTLD_NEXT, "GPTLreset_timer");
    }
    if(GPTLreset_timer_h != NULL) {
        result = (*GPTLreset_timer_h)(name);
    }

    TAU_GPTL_UNIMPLEMENTED();
    return result;
}

int GPTLfinalize (void) {
    TAU_GPTL_LOG();
    int result = 0;
    Tau_profile_exit_all_threads();
    Tau_destructor_trigger();

    static int (*GPTLfinalize_h)(void) = NULL;
    if(GPTLfinalize_h == NULL) {
        GPTLfinalize_h = dlsym(RTLD_NEXT, "GPTLfinalize");
    }
    if(GPTLfinalize_h != NULL) {
        result = (*GPTLfinalize_h)();
    }

    return result;
}

int GPTLget_memusage (float *usage) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLget_memusage_h)(float * usage) = NULL;
    if(GPTLget_memusage_h == NULL) {
        GPTLget_memusage_h = dlsym(RTLD_NEXT, "GPTLget_memusage");
    }
    if(GPTLget_memusage_h != NULL) {
        result = (*GPTLget_memusage_h)(usage);
    }

    return result;
}

int GPTLprint_memusage (const char * str) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLprint_memusage_h)(const char * str) = NULL;
    if(GPTLprint_memusage_h == NULL) {
        GPTLprint_memusage_h = dlsym(RTLD_NEXT, "GPTLprint_memusage");
    }
    if(GPTLprint_memusage_h != NULL) {
        result = (*GPTLprint_memusage_h)(str);
    }

    return result;
}

int GPTLprint_rusage (const char * str) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLprint_rusage_h)(const char * str) = NULL;
    if(GPTLprint_rusage_h == NULL) {
        GPTLprint_rusage_h = dlsym(RTLD_NEXT, "GPTLprint_rusage");
    }
    if(GPTLprint_rusage_h != NULL) {
        result = (*GPTLprint_rusage_h)(str);
    }

    return result;
}                                      

int GPTLget_procsiz (float * procsiz_out, float * rss_out) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLget_procsiz_h)(float * procsiz_out, float * rss_out) = NULL;
    if(GPTLget_procsiz_h == NULL) {
        GPTLget_procsiz_h = dlsym(RTLD_NEXT, "GPTLget_procsiz");
    }
    if(GPTLget_procsiz_h != NULL) {
        result = (*GPTLget_procsiz_h)(procsiz_out, rss_out);
    }

    return result;
} 

int GPTLenable (void) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLenable_h)(void) = NULL;
    if(GPTLenable_h == NULL) {
        GPTLenable_h = dlsym(RTLD_NEXT, "GPTLenable");
    }
    if(GPTLenable_h != NULL) {
        result = (*GPTLenable_h)();
    }

    gptl_enabled = true;
    return result;
} 

int GPTLdisable (void) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLdisable_h)(void) = NULL;
    if(GPTLdisable_h == NULL) {
        GPTLdisable_h = dlsym(RTLD_NEXT, "GPTLdisable");
    }
    if(GPTLdisable_h != NULL) {
        result = (*GPTLdisable_h)();
    }

    gptl_enabled = false;
    return result;
} 

int GPTLsetutr (const int option) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLsetutr_h)(const int option) = NULL;
    if(GPTLsetutr_h == NULL) {
        GPTLsetutr_h = dlsym(RTLD_NEXT, "GPTLsetutr");
    }
    if(GPTLsetutr_h != NULL) {
        result = (*GPTLsetutr_h)(option);
    }

    return result;
} 

int GPTLquery (const char *name, int t, int *count, int *onflg, double *wallclock,
        double *dusr, double *dsys, long long *papicounters_out, const int maxcounters) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLquery_h)(const char *name, int t, int *count, int *onflg, double *wallclock,
        double *dusr, double *dsys, long long *papicounters_out, const int maxcounters) = NULL;
    if(GPTLquery_h == NULL) {
        GPTLquery_h = dlsym(RTLD_NEXT, "GPTLquery");
    }
    if(GPTLquery_h != NULL) {
        result = (*GPTLquery_h)(name, t, count, onflg, wallclock, dusr, dsys, papicounters_out, maxcounters);
    }

    return result;
} 

int GPTLget_wallclock (const char *timername, int t, double *value)  {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLget_wallclock_h)(const char * timername, int t, double * value) = NULL;
    if(GPTLget_wallclock_h == NULL) {
        GPTLget_wallclock_h = dlsym(RTLD_NEXT, "GPTLget_wallclock");
    }
    if(GPTLget_wallclock_h != NULL) {
        result = (*GPTLget_wallclock_h)(timername, t, value);
    }

    return result;
} 

int GPTLget_wallclock_latest (const char * timername, int t, double *value) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLget_wallclock_latest_h)(const char * timername, int t, double * value) = NULL;
    if(GPTLget_wallclock_latest_h == NULL) {
        GPTLget_wallclock_latest_h = dlsym(RTLD_NEXT, "GPTLget_wallclock_latest");
    }
    if(GPTLget_wallclock_latest_h != NULL) {
        result = (*GPTLget_wallclock_latest_h)(timername, t, value);
    }

    return result;
} 

int GPTLget_threadwork (const char *name, double *maxwork, double *imbal)   {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLget_threadwork_h)(const char * name, double * maxwork, double * imbal) = NULL;
    if(GPTLget_threadwork_h == NULL) {
        GPTLget_threadwork_h = dlsym(RTLD_NEXT, "GPTLget_threadwork");
    }
    if(GPTLget_threadwork_h != NULL) {
        result = (*GPTLget_threadwork_h)(name, maxwork, imbal);
    }

    return result;
} 

int GPTLstartstop_val (const char *name, double value)   {
    TAU_GPTL_LOG_NAME(name);
    int result = 0;

    static int (*GPTLstartstop_val_h)(const char * name, double value) = NULL;
    if(GPTLstartstop_val_h == NULL) {
        GPTLstartstop_val_h = dlsym(RTLD_NEXT, "GPTLstartstop_val");
    }
    if(GPTLstartstop_val_h != NULL) {
        result = (*GPTLstartstop_val_h)(name, value);
    }

    if(gptl_enabled) {
        gptl_tau_increment(name, value, 1);
    }
    return result;
} 

int GPTLget_nregions (int t, int * nregions)  {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLget_nregions_h)(int t, int * nregions) = NULL;
    if(GPTLget_nregions_h == NULL) {
        GPTLget_nregions_h = dlsym(RTLD_NEXT, "GPTLget_nregions");
    }
    if(GPTLget_nregions_h != NULL) {
        result = (*GPTLget_nregions_h)(t, nregions);
    }

    return result;
} 

int GPTLget_regionname (int t, int region, char * name, int nc)   {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLget_regionname_h)(int t, int region, char * name, int nc) = NULL;
    if(GPTLget_regionname_h == NULL) {
        GPTLget_regionname_h = dlsym(RTLD_NEXT, "GPTLget_regionname");
    }
    if(GPTLget_regionname_h != NULL) {
        result = (*GPTLget_regionname_h)(t, region, name, nc);
    }

    return result;
} 

int GPTL_PAPIlibraryinit (void)   {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTL_PAPIlibraryinit_h)(void) = NULL;
    if(GPTL_PAPIlibraryinit_h == NULL) {
        GPTL_PAPIlibraryinit_h = dlsym(RTLD_NEXT, "GPTL_PAPIlibraryinit");
    }
    if(GPTL_PAPIlibraryinit_h != NULL) {
        result = (*GPTL_PAPIlibraryinit_h)();
    }

    return result;
} 

int GPTLevent_name_to_code (const char *name, int *code)   {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLevent_name_to_code_h)(const char * name, int * code) = NULL;
    if(GPTLevent_name_to_code_h == NULL) {
        GPTLevent_name_to_code_h = dlsym(RTLD_NEXT, "GPTLevent_name_to_code");
    }
    if(GPTLevent_name_to_code_h != NULL) {
        result = (*GPTLevent_name_to_code_h)(name, code);
    }

    return result;
} 

int GPTLevent_code_to_name (const int code, char * name)   {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLevent_code_to_name_h)(const int code, char * name) = NULL;
    if(GPTLevent_code_to_name_h == NULL) {
        GPTLevent_code_to_name_h = dlsym(RTLD_NEXT, "GPTLevent_code_to_name");
    }
    if(GPTLevent_code_to_name_h != NULL) {
        result = (*GPTLevent_code_to_name_h)(code, name);
    }

    return result;
} 

int GPTLget_eventvalue (const char *timername, const char *eventname, int t, double *value)   {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLget_eventvalue_h)(const char * timername, const char * eventname, double * value) = NULL;
    if(GPTLget_eventvalue_h == NULL) {
        GPTLget_eventvalue_h = dlsym(RTLD_NEXT, "GPTLget_eventvalue");
    }
    if(GPTLget_eventvalue_h != NULL) {
        result = (*GPTLget_eventvalue_h)(timername, eventname, value);
    }

    return result;
} 

int GPTLnum_errors (void) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLnum_errors_h)(void) = NULL;
    if(GPTLnum_errors_h == NULL) {
        GPTLnum_errors_h = dlsym(RTLD_NEXT, "GPTLnum_errors");
    }
    if(GPTLnum_errors_h != NULL) {
        result = (*GPTLnum_errors_h)();
    }

    return result;
} 

int GPTLnum_warn (void) {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLnum_warn_h)(void) = NULL;
    if(GPTLnum_warn_h == NULL) {
        GPTLnum_warn_h = dlsym(RTLD_NEXT, "GPTLnum_warn");
    }
    if(GPTLnum_warn_h != NULL) {
        result = (*GPTLnum_warn_h)();
    }

    return result;
} 

int GPTLget_count (const char *timername, int t, int *count)   {
    TAU_GPTL_LOG();
    int result = 0;

    static int (*GPTLget_count_h)(const char * timername, int t, int * count) = NULL;
    if(GPTLget_count_h == NULL) {
        GPTLget_count_h = dlsym(RTLD_NEXT, "GPTLget_count");
    }
    if(GPTLget_count_h != NULL) {
        result = (*GPTLget_count_h)(timername, t, count);
    }

    return result;
} 

// E3SM-specific functions

int GPTLprefix_set(const char * prefixname) {
    TAU_GPTL_LOG_NAME(prefixname);
    int result = 0;

    static int (*GPTLprefix_set_h)(const char * prefixname) = NULL;
    if(GPTLprefix_set_h == NULL) {
        GPTLprefix_set_h = dlsym(RTLD_NEXT, "GPTLprefix_set");
    }
    if(GPTLprefix_set_h != NULL) {
        result = (*GPTLprefix_set_h)(prefixname);
    }

    if(gptl_prefix != NULL) {
        free(gptl_prefix);
    }
    gptl_prefix = strdup(prefixname);

    return result;
}

int GPTLprefix_setf(const char * prefixname, const int prefixlen) {
    int result = 0;

    static int (*GPTLprefix_setf_h)(const char * prefixname, const int prefixlen) = NULL;
    if(GPTLprefix_setf_h == NULL) {
        GPTLprefix_setf_h = dlsym(RTLD_NEXT, "GPTLprefix_setf");
    }
    if(GPTLprefix_setf_h != NULL) {
        result = (*GPTLprefix_setf_h)(prefixname, prefixlen);
    }

    if(gptl_prefix != NULL) {
        free(gptl_prefix);
    }

    char * nullterm_name = (char *)malloc(prefixlen + 1);
    memcpy(nullterm_name, prefixname, prefixlen);
    nullterm_name[prefixlen] = '\0';
    TAU_GPTL_LOG_NAME(nullterm_name);
    gptl_prefix = nullterm_name;
    return result;
}

int GPTLprefix_unset(void) {
    int result = 0;

    static int (*GPTLprefix_unset_h)(void) = NULL;
    if(GPTLprefix_unset_h == NULL) {
        GPTLprefix_unset_h = dlsym(RTLD_NEXT, "GPTLprefix_unset");
    }
    if(GPTLprefix_unset_h != NULL) {
        result = (*GPTLprefix_unset_h)();
    }

    if(gptl_prefix != NULL) {
        free(gptl_prefix);
    }

    gptl_prefix = NULL;

    return result;
}

int GPTLstartf(const char * timername, const int namelen) {
    int result = 0;

    static int (*GPTLstartf_h)(const char * timername, const int namelen) = NULL;
    if(GPTLstartf_h == NULL) {
        GPTLstartf_h = dlsym(RTLD_NEXT, "GPTLstartf");
    }
    if(GPTLstartf_h != NULL) {
        result = (*GPTLstartf_h)(timername, namelen);
    }

    if(gptl_enabled) {
        char nullterm_name[namelen + 1];
        memcpy(nullterm_name, timername, namelen);
        nullterm_name[namelen] = '\0';
        TAU_GPTL_LOG_NAME(nullterm_name);
        gptl_tau_start(nullterm_name);
    }

    return result;
}

int GPTLstartf_handle (const char * timername, const int namelen, void ** handle) {
    int result = 0;

    static int (*GPTLstartf_handle_h)(const char * timername, const int namelen, void ** handle) = NULL;
    if(GPTLstartf_handle_h == NULL) {
        GPTLstartf_handle_h = dlsym(RTLD_NEXT, "GPTLstartf_handle");
    }
    if(GPTLstartf_handle_h != NULL) {
        result = (*GPTLstartf_handle_h)(timername, namelen, handle);
    }

    if(gptl_enabled) {
        char nullterm_name[namelen + 1];
        memcpy(nullterm_name, timername, namelen);
        nullterm_name[namelen] = '\0';
        TAU_GPTL_LOG_NAME(nullterm_name);
        gptl_tau_start(nullterm_name);
    }

    return result;
}


int GPTLstopf (const char * timername, const int namelen) {
    int result = 0;

    static int (*GPTLstopf_h)(const char * timername, const int namelen) = NULL;
    if(GPTLstopf_h == NULL) {
        GPTLstopf_h = dlsym(RTLD_NEXT, "GPTLstopf");
    }
    if(GPTLstopf_h != NULL) {
        result = (*GPTLstopf_h)(timername, namelen);
    }

    if(gptl_enabled) {
        char nullterm_name[namelen + 1];
        memcpy(nullterm_name, timername, namelen);
        nullterm_name[namelen] = '\0';
        TAU_GPTL_LOG_NAME(nullterm_name);
        gptl_tau_stop(nullterm_name);
    }

    return result;

}


int GPTLstopf_handle (const char * timername, const int namelen, void ** handle) {
    int result = 0;

    static int (*GPTLstopf_handle_h)(const char * timername, const int namelen, void ** handle) = NULL;
    if(GPTLstopf_handle_h == NULL) {
        GPTLstopf_handle_h = dlsym(RTLD_NEXT, "GPTLstopf_handle");
    }
    if(GPTLstopf_handle_h != NULL) {
        result = (*GPTLstopf_handle_h)(timername, namelen, handle);
    }

    if(gptl_enabled) {
        char nullterm_name[namelen + 1];
        memcpy(nullterm_name, timername, namelen);
        nullterm_name[namelen] = '\0';
        TAU_GPTL_LOG_NAME(nullterm_name);
        gptl_tau_stop(nullterm_name);
    }

    return result;
}

int GPTLstartstop_vals (const char * timername, double add_time, int add_count) {
    TAU_GPTL_LOG_NAME(timername);
    int result = 0;

    static int (*GPTLstartstop_vals_h)(const char * timername, double add_time, int add_count)  = NULL;
    if(GPTLstartstop_vals_h == NULL) {
        GPTLstartstop_vals_h = dlsym(RTLD_NEXT, "GPTLstartstop_vals");
    }
    if(GPTLstartstop_vals_h != NULL) {
        result = (*GPTLstartstop_vals_h)(timername, add_time, add_count);
    }

    if(gptl_enabled) {
        gptl_tau_increment(timername, add_time, add_count);
    }
    return result;
}

int GPTLstartstop_valsf (const char * timername, const int namelen, double add_time, int add_count) {
    int result = 0;

    static int (*GPTLstartstop_valsf_h)(const char * timername, const int namelen, double add_time, int add_count)  = NULL;
    if(GPTLstartstop_valsf_h == NULL) {
        GPTLstartstop_valsf_h = dlsym(RTLD_NEXT, "GPTLstartstop_valsf");
    }
    if(GPTLstartstop_valsf_h != NULL) {
        result = (*GPTLstartstop_valsf_h)(timername, namelen, add_time, add_count);
    }

    if(gptl_enabled) {
        char nullterm_name[namelen + 1];
        memcpy(nullterm_name, timername, namelen);
        nullterm_name[namelen] = '\0';
        TAU_GPTL_LOG_NAME(nullterm_name);
        gptl_tau_increment(nullterm_name, add_time, add_count);
    }
    return result;

}

