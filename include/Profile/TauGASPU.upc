/*   $Source: /var/local/cvs/upcr/profile/dump/gaspu.upc,v $ */
/*      $Date: 2007/01/25 18:50:17 $ */
/*  $Revision: 1.8 $ */
/*  Description: upcalls from GASP instrumentation tool into UPC code */
/*  Copyright 2005, Dan Bonachea <bonachea@cs.berkeley.edu> */

#include <upc.h>
#ifdef __UPC_COLLECTIVE__
#include <upc_collective.h>
#endif
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* NOTE: this file is currently compiled with every instrumented application, 
   therefore its contents should be kept to a minimum */

/* disable instrumentation in this file, if possible */
#pragma pupc off

#ifdef __BERKELEY_UPC__
/* ensure code in this file does not disturb line numbering */
#pragma UPCR NO_SRCPOS
#endif

typedef uint64_t gasp_tick_t;

int gaspu_zero = 0;
void gaspu_init(int *pmythread, int *pthreads)
{
    *pmythread = MYTHREAD;
    *pthreads = THREADS;
    if (gaspu_zero)
        gasp_init(0, 0, 0); /* this is never called - just used to force libgasp linkage */
}

void gaspu_barrier() {
    upc_barrier;
}

#ifdef __BERKELEY_UPC_RUNTIME__
void gaspu_ticks_now(gasp_tick_t *pval) {
    *pval = (gasp_tick_t)bupc_ticks_now();
}
void gaspu_ticks_to_sec(gasp_tick_t ticks, double *pval) {
    *pval = bupc_ticks_to_ns((bupc_tick_t)ticks)/1E9;
}
#else /* assume Berkeley timers unavailable */
#include <time.h>
#include <sys/time.h>
void gaspu_ticks_now(gasp_tick_t *pval) {
    struct timeval tv;
#if defined(_CRAY) || defined(_UNICOSMP)
    retry:
#endif
    if (gettimeofday(&tv, NULL)) {
        perror("gettimeofday");
        {
            return;
        }
    }
    *pval = ((int64_t) tv.tv_sec) * 1000000 + tv.tv_usec;
#if defined(_CRAY) || defined(_UNICOSMP)
    /* fix an empirically observed bug in UNICOS gettimeofday(),
     which occasionally returns ridiculously incorrect values
     */
    if(*pval < (((int64_t)3) << 48)) goto retry;
#endif
}
void gaspu_ticks_to_sec(gasp_tick_t ticks, double *pval) {
    *pval = ticks / 1E6;
}
#endif

void gaspu_dump_shared(void *ptr_to_ptr_to_shared, char *outputbuf, int bufsz) {
    shared void *pts = *(shared void **) ptr_to_ptr_to_shared;
#ifdef __BERKELEY_UPC__
    if (bupc_dump_shared(pts, outputbuf, bufsz)) perror("bupc_dump_shared");
#else
    snprintf(outputbuf, bufsz, "<addrfield=%p thread=%d phase=%d>",
            (void *) upc_addrfield(pts), (int) upc_threadof(pts),
            (int) upc_phaseof(pts));
#endif
}

void gaspu_getenv(const char *key, const char **val) {
    *val = getenv(key);
}

typedef struct {
    int flag;
    const char *name;
} flaginfo_t;
#define FLAG(name) { name, #name }

void gaspu_flags_to_string(int flags, char *str, int sz) {
    char *p = str;
    static flaginfo_t known_flags[] = {
#ifdef UPC_IN_ALLSYNC
            FLAG(UPC_IN_ALLSYNC),
            FLAG(UPC_IN_MYSYNC),
            FLAG(UPC_IN_NOSYNC),
            FLAG(UPC_OUT_ALLSYNC),
            FLAG(UPC_OUT_MYSYNC),
            FLAG(UPC_OUT_NOSYNC),
#else
            { -1, "UNKNOWN" },
#endif
            { 0, 0 } };
    for (int i = 0; known_flags[i].name && sz > 1; i++) {
        if (known_flags[i].flag & flags) {
            int v;
            if (p > str && sz > 2) {
                strcat(p, "|");
                p++;
                sz--;
            }
            strncpy(p, known_flags[i].name, sz);
            p[sz - 1] = '\0';
            v = strlen(p);
            sz -= v;
            p += v;
        }
    }
}

void gaspu_collop_to_string(int op, char *str, int sz) {
#ifdef __UPC_COLLECTIVE__
    static flaginfo_t known_flags[] = {
        FLAG(UPC_ADD),
        FLAG(UPC_MULT),
        FLAG(UPC_AND),
        FLAG(UPC_OR),
        FLAG(UPC_XOR),
        FLAG(UPC_LOGAND),
        FLAG(UPC_LOGOR),
        FLAG(UPC_MIN),
        FLAG(UPC_MAX),
        FLAG(UPC_FUNC),
        FLAG(UPC_NONCOMM_FUNC),
        {   0, 0}
    };
    for (int i=0; known_flags[i].name; i++) {
        if (known_flags[i].flag == op) {
            strncpy(str, known_flags[i].name, sz);
            str[sz-1] = '\0';
            {   return;}
        }
    }
#endif
    strncpy(str, "<unknown upc_op_t>", sz);
    str[sz - 1] = '\0';
}

int gaspu_upcall_threadof(void *ptr) {
    shared void * p = *(shared void **)ptr;
    return (int)upc_threadof(p);
}
