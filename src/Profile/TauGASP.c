/*   $Source: /mnt/fast/tau2git/cvsroot/tau2/src/Profile/TauGASP.c,v $ */
/*      $Date: 2009/11/07 09:38:23 $ */
/*  $Revision: 1.1 $ */
/*  Description: main implementation of the GASP-based tracing tool 'dump' */
/*  Copyright 2005, Dan Bonachea <bonachea@cs.berkeley.edu> */

#include <Profile/Profiler.h>
#include <Profile/TauEnv.h>
#include <Profile/TauAPI.h>

#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

#include <gasp.h>
#include <gasp_upc.h>

#include <Profile/TauGASP.h>

/* disable instrumentation in this file, if possible */
#pragma pupc off

#ifdef __BERKELEY_UPC__
/* ensure code in this file does not disturb line numbering */
#pragma UPCR NO_SRCPOS
#endif

/* internal tool events, placed at end of user event range */
#define GASPI_EVT_BASE          (GASP_UPC_USEREVT_END - GASPI_RESERVEDEVTS)
#define GASPI_INIT              GASPI_EVT_BASE+0
#define GASPI_CREATE_EVENT      GASPI_EVT_BASE+1
#define GASPI_CONTROL           GASPI_EVT_BASE+2
#define GASPI_RESERVEDEVTS      3

/* number of user events */
#define GASPI_UPC_USEREVT_NUM ((GASP_UPC_USEREVT_END-GASPI_RESERVEDEVTS)-GASP_UPC_USEREVT_START+1)

static int tau_upc_tagid_f = 0;
#define TAU_UPC_TAGID (tau_upc_tagid_f = (tau_upc_tagid_f & 255))
#define TAU_UPC_TAGID_NEXT ((++tau_upc_tagid_f) & 255)

typedef struct {
    const char *name;
    const char *desc;
} gasp_userevt_t;

struct _gasp_context_S {
    int inupcall;
    int enabled;
    int forceflush; /* force flush after each output (useful for crashes) */
    int skipoutput; /* fully parse events but skip the final output to file (for debugging tool) */
    gasp_model_t srcmodel;
    int mythread;
    int threads;
    gasp_userevt_t *userevt;
    size_t userevt_cnt;
    size_t userevt_sz;
};

/* make a reentrant-safe upcall into UPC */
#define GASPI_UPCALL(context, fncall) do { \
  assert(!context->inupcall);              \
  context->inupcall = 1;                   \
  fncall;                                  \
  context->inupcall = 0;                   \
} while(0)

int gasp_control(gasp_context_t context, int on) {
    int oldval = context->enabled;
    int newval = !!on;
    if (oldval ^ newval) { /* control transition */
        context->enabled = 1;
        gasp_event_notify(context, GASPI_CONTROL, GASP_ATOMIC, NULL, 0, 0, newval);
        context->enabled = newval;
    }
    return oldval;
}

void gaspi_err(const char *fmt, ...) {
    char buf[1024];
    va_list argptr;
    va_start(argptr, fmt);
    /*  pass in last argument */
    vsnprintf(buf, 1024, fmt, argptr);
    va_end(argptr);
    fprintf(stderr, "*** GASP FATAL ERROR: %s\n", buf);
    abort();
}

int gaspi_getenvYN(gasp_context_t context, const char *key, int defaultval) {
    const char *val = NULL;
    GASPI_UPCALL(context, gaspu_getenv(key, &val));
    if (val) {
        return (atoi(val) || *val == 'y' || *val == 'Y');
    }
    return defaultval;
}

unsigned int gasp_create_event(gasp_context_t context, const char *name, const char *desc) {
    int idx = context->userevt_cnt;
    int retval;
    if (context->userevt_cnt == context->userevt_sz) {
        if (context->userevt_cnt == GASPI_UPC_USEREVT_NUM)
            gaspi_err("gasp_create_event(): too many user events. Max is %i",
                      GASPI_UPC_USEREVT_NUM);
        context->userevt_sz = (context->userevt_sz * 2) + 1;
        context->userevt = realloc(context->userevt,
                sizeof(gasp_userevt_t) * context->userevt_sz);
        assert(context->userevt);
    }
    context->userevt[idx].name = (name ? strdup(name) : "USER_EVENT");
    context->userevt[idx].desc = (desc ? strdup(desc) : "");
    context->userevt_cnt++;
    retval = idx + GASP_UPC_USEREVT_START;

    gasp_event_notify(context, GASPI_CREATE_EVENT, GASP_ATOMIC, NULL, 0, 0, name, desc, retval);
    return retval;
}


gasp_context_t gasp_init(gasp_model_t srcmodel, int *argc, char ***argv) {
    int nodeid;

    /* allocate a local context */
    gasp_context_t context = (gasp_context_t) calloc(1, sizeof(struct _gasp_context_S));
    assert(context->srcmodel == GASP_MODEL_UPC);
    /* for now */
    context->srcmodel = srcmodel;
    gaspu_init(&(context->mythread), &(context->threads));
    context->enabled = 1;

    /* query system parameters */
    context->forceflush = gaspi_getenvYN(context, "GASP_FLUSH", 0);

    Tau_create_top_level_timer_if_necessary();

    if (TauEnv_get_ebs_enabled()) {
        Tau_sampling_init_if_necessary();
    }

    Tau_signal_initialization();

#ifdef TAU_MONITORING
    Tau_mon_connect();
#endif /* TAU_MONITORING */

#ifdef TAU_BGP
    if (TauEnv_get_ibm_bg_hwp_counters()) {
        int upcErr;
        Tau_Bg_hwp_counters_start(&upcErr);
        if (upcErr != 0) {
            printf("TAU ERROR: ** Error starting IBM BGP UPC hardware performance counters\n");
        }
    }
#endif /* TAU_BGP */

    nodeid = TAU_PROFILE_GET_NODE();
    if (nodeid == -1) {
        tau_totalnodes(1,context->threads);
        TAU_PROFILE_SET_NODE(context->mythread);
    }

#ifdef TAU_MPI
    if (TauEnv_get_synchronize_clocks()) {
        TauSyncClocks();
    }
#endif

    return context;
}

void gasp_event_notify(gasp_context_t context, unsigned int evttag,
        gasp_evttype_t evttype, const char *filename, int linenum, int colnum, ...) {
    va_list argptr;
    va_start(argptr, colnum);
    /*  pass in last argument */
    gasp_event_notifyVA(context, evttag, evttype, filename, linenum, colnum, argptr);
    va_end(argptr);
}

void gasp_event_notifyVA(gasp_context_t context, unsigned int evttag,
        gasp_evttype_t evttype, const char *filename, int linenum, int colnum,
        va_list argptr) {
    gasp_tick_t curtime;
    const char *typestr = "<unknown type>";
    const char *tagstr = "<unknown tag>";
    const char *argstr = "";
    int is_user_evt = 0;

    assert(context);
    if (!context->enabled)
        return; /* disabled by control */
    if (context->inupcall)
        return; /* reentrant call */

    GASPI_UPCALL(context, gaspu_ticks_now(&curtime));
    /* get current time */

    switch (evttype) {
    case GASP_START:  typestr = "START";  break;
    case GASP_END:    typestr = "END";    break;
    case GASP_ATOMIC: typestr = "ATOMIC"; break;
    }

#define _GASPI_TAG(prefix,tag,args,resultargs)          \
    case prefix##tag: tagstr=#tag;                      \
              if (evttype==GASP_END) argstr=resultargs; \
              else argstr=args;                         \
              break
#define GASPI_TAG(tag,args,resultargs) _GASPI_TAG(GASP_,tag,args,resultargs)
#define GASPI_TAG_INTERNAL(tag,args,resultargs) _GASPI_TAG(GASPI_,tag,args,resultargs)

#define TAU_TRACE_GASPI_GET(src, n)  { \
    int remote_rank; \
    GASPI_UPCALL(context, remote_rank=gaspu_upcall_threadof(src)); \
    switch(evttype) { \
    case GASP_START: { TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, context->mythread, n, remote_rank); } break; \
    case GASP_END: { TAU_TRACE_RECVMSG(TAU_UPC_TAGID, remote_rank, n); } break; \
    case GASP_ATOMIC: { TAU_REGISTER_EVENT(variable, tagstr); TAU_EVENT(variable, n); } break; \
    } } while(0)

#define TAU_TRACE_GASPI_PUT(dst, n) do { \
    int remote_rank; \
    GASPI_UPCALL(context, remote_rank=gaspu_upcall_threadof(dst)); \
    switch(evttype) { \
    case GASP_START: { TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, remote_rank, n); } break; \
    case GASP_END: { TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, context->mythread, n, remote_rank); } break; \
    case GASP_ATOMIC: { TAU_REGISTER_EVENT(variable, tagstr); TAU_EVENT(variable, n); } break; \
    } } while(0)

    switch (evttag) {
    /* define each event and its args - see arg definitions below */

    /* internal tool events */
    GASPI_TAG_INTERNAL(INIT, "s", "");
    GASPI_TAG_INTERNAL(CREATE_EVENT, "SSI", "");
    GASPI_TAG_INTERNAL(CONTROL, "I", "");

    /* system events */
    #ifdef GASP_UPC_ALL_ALLOC
        GASPI_TAG(UPC_ALL_ALLOC, "zz", "zzP");
    #endif
    #ifdef GASP_UPC_GLOBAL_ALLOC
        GASPI_TAG(UPC_GLOBAL_ALLOC, "zz", "zzP");
    #endif
    #ifdef GASP_UPC_ALLOC
        GASPI_TAG(UPC_ALLOC, "z", "zP");
    #endif
    #ifdef GASP_UPC_FREE
        GASPI_TAG(UPC_FREE, "P", "P");
    #endif

    #ifdef GASP_BUPC_STATIC_SHARED
    #if GASP_UPC_VERSION >= 0x020311
        GASPI_TAG(BUPC_STATIC_SHARED,"zzPpzss","");
    #else
        GASPI_TAG(BUPC_STATIC_SHARED,"zzPp","");
    #endif
    #endif

    #ifdef GASP_C_FUNC
        GASPI_TAG(C_FUNC, "s", "s");
    #endif
    #ifdef GASP_C_MALLOC
        GASPI_TAG(C_MALLOC,"z","zp");
    #endif
    #ifdef GASP_C_REALLOC
        GASPI_TAG(C_REALLOC,"pz","pzp");
    #endif
    #ifdef GASP_C_FREE
        GASPI_TAG(C_FREE,"p","p");
    #endif

    #ifdef GASP_UPC_ALL_LOCK_ALLOC
        GASPI_TAG(UPC_ALL_LOCK_ALLOC, "", "K");
    #endif
    #ifdef GASP_UPC_GLOBAL_LOCK_ALLOC
        GASPI_TAG(UPC_GLOBAL_LOCK_ALLOC, "", "K");
    #endif
    #ifdef GASP_UPC_LOCK_FREE
        GASPI_TAG(UPC_LOCK_FREE, "K", "K");
    #endif

    #ifdef GASP_UPC_LOCK
        GASPI_TAG(UPC_LOCK, "K", "K");
    #endif
    #ifdef GASP_UPC_LOCK_ATTEMPT
        GASPI_TAG(UPC_LOCK_ATTEMPT, "K", "KI");
    #endif
    #ifdef GASP_UPC_UNLOCK
        GASPI_TAG(UPC_UNLOCK, "K", "K");
    #endif

    #ifdef GASP_UPC_NOTIFY
        GASPI_TAG(UPC_NOTIFY, "II", "II");
    #endif
    #ifdef GASP_UPC_WAIT
        GASPI_TAG(UPC_WAIT, "II", "II");
    #endif
    #ifdef GASP_UPC_BARRIER
        GASPI_TAG(UPC_BARRIER, "II", "II");
    #endif
    #ifdef GASP_UPC_FENCE
        GASPI_TAG(UPC_FENCE, "II", "II");
    #endif

    #ifdef GASP_UPC_MEMGET
        GASPI_TAG(UPC_MEMGET, "pPz", "pPz");
    #endif
    #ifdef GASP_UPC_MEMPUT
        GASPI_TAG(UPC_MEMPUT, "Ppz", "Ppz");
    #endif
    #ifdef GASP_UPC_MEMCPY
        GASPI_TAG(UPC_MEMCPY, "PPz", "PPz");
    #endif
    #ifdef GASP_UPC_MEMSET
        GASPI_TAG(UPC_MEMSET, "PCz", "PCz");
    #endif

    #ifdef GASP_UPC_PUT
        GASPI_TAG(UPC_PUT, "IPpz", "IPpz");
    #endif
    #ifdef GASP_UPC_GET
        GASPI_TAG(UPC_GET, "IpPz", "IpPz");
    #endif
    #ifdef GASP_UPC_NB_GET_INIT
        GASPI_TAG(UPC_NB_GET_INIT,"IpPz","IpPzH");
    #endif
    #ifdef GASP_UPC_NB_PUT_INIT
        GASPI_TAG(UPC_NB_PUT_INIT,"IPpz","IPpzH");
    #endif
    #ifdef GASP_UPC_NB_SYNC
        GASPI_TAG(UPC_NB_SYNC,"H","H");
    #endif
    #ifdef GASP_BUPC_NB_TRYSYNC
        GASPI_TAG(BUPC_NB_TRYSYNC,"H","HI");
    #endif

    #ifdef GASP_BUPC_NB_MEMGET_INIT
        GASPI_TAG(BUPC_NB_MEMGET_INIT,"pPz","pPzH");
    #endif
    #ifdef GASP_BUPC_NB_MEMPUT_INIT
        GASPI_TAG(BUPC_NB_MEMPUT_INIT,"Ppz","PpzH");
    #endif
    #ifdef GASP_BUPC_NB_MEMCPY_INIT
        GASPI_TAG(BUPC_NB_MEMCPY_INIT,"PPz","PPzH");
    #endif

    #ifdef GASP_UPC_COLLECTIVE_EXIT
        GASPI_TAG(UPC_COLLECTIVE_EXIT, "I", "I");
    #endif
    #ifdef GASP_UPC_NONCOLLECTIVE_EXIT
        GASPI_TAG(UPC_NONCOLLECTIVE_EXIT, "I", "I");
    #endif

    #ifdef GASP_UPC_ALL_BROADCAST
        GASPI_TAG(UPC_ALL_BROADCAST, "PPzF","PPzF");
    #endif
    #ifdef GASP_UPC_ALL_SCATTER
        GASPI_TAG(UPC_ALL_SCATTER, "PPzF","PPzF");
    #endif
    #ifdef GASP_UPC_ALL_GATHER
        GASPI_TAG(UPC_ALL_GATHER, "PPzF","PPzF");
    #endif
    #ifdef GASP_UPC_ALL_GATHER_ALL
        GASPI_TAG(UPC_ALL_GATHER_ALL,"PPzF","PPzF");
    #endif
    #ifdef GASP_UPC_ALL_EXCHANGE
        GASPI_TAG(UPC_ALL_EXCHANGE, "PPzF","PPzF");
    #endif
    #ifdef GASP_UPC_ALL_PERMUTE
        GASPI_TAG(UPC_ALL_PERMUTE, "PPPzF","PPPzF");
    #endif

    #ifdef GASP_UPC_ALL_REDUCE
        GASPI_TAG(UPC_ALL_REDUCE, "PPOzzpFR","PPOzzpFR");
    #endif
    #ifdef GASP_UPC_ALL_PREFIX_REDUCE
        GASPI_TAG(UPC_ALL_PREFIX_REDUCE,"PPOzzpFR","PPOzzpFR");
    #endif

    #ifdef GASP_UPC_FORALL
        /* TODO: GASP_UPC_FORALL support */
    #endif

    #ifdef GASP_UPC_NB_GET_DATA
        /* TODO: GASP_UPC_NB_GET_DATA support */
    #endif
    #ifdef GASP_UPC_NB_PUT_DATA
        /* TODO: GASP_UPC_NB_PUT_DATA support */
    #endif

    #ifdef GASP_UPC_CACHE_MISS
        /* TODO: GASP_UPC_CACHE_MISS support */
    #endif
    #ifdef GASP_UPC_CACHE_HIT
        /* TODO: GASP_UPC_CACHE_HIT support */
    #endif
    #ifdef GASP_UPC_CACHE_INVALIDATE
        /* TODO: GASP_UPC_CACHE_INVALIDATE support */
    #endif

    default:
        if (evttag >= GASP_UPC_USEREVT_START && evttag <= GASP_UPC_USEREVT_END) {
            /* it's a user event */
            int id = evttag - GASP_UPC_USEREVT_START;
            if (id < context->userevt_cnt) {
                tagstr = context->userevt[id].name;
                argstr = context->userevt[id].desc;
                is_user_evt = 1;
            } else {
                fprintf(stderr, "ERROR: id=%d < userevt_cnt=%d.  "
                        "Check that %s was compiled with UPC compiler.\n",
                        id, context->userevt_cnt, __FILE__);
                fflush(stderr);
            }
        }
    }

    if(evttype == GASP_START) {
        /* Start TAU timing before processing trace events */
        TAU_START(tagstr);
    }

    switch(evttag) {

#ifdef GASP_BUPC_STATIC_SHARED
    case GASP_BUPC_STATIC_SHARED:
    {
        if (evttype == GASP_ATOMIC) {
            int nblocks = (int)va_arg(argptr, int);
            int nbytes = (int)va_arg(argptr, int);
            TAU_REGISTER_EVENT(variable, tagstr);
            TAU_EVENT(variable, nbytes);
        }
        break;
    }
#endif

    case GASP_UPC_GET:
    {
        int relaxed1 = (int)va_arg(argptr, int);
        void *dst = (void *)va_arg(argptr, void *);
        gasp_upc_PTS_t *src = (gasp_upc_PTS_t *) va_arg(argptr, gasp_upc_PTS_t *);
        size_t n = (int) va_arg(argptr, int);
        TAU_TRACE_GASPI_GET(src, n);
        break;
    }
    case GASP_UPC_PUT:
    {
        int relaxed2 = (int) va_arg(argptr, int);
        gasp_upc_PTS_t *dst = (gasp_upc_PTS_t *) va_arg(argptr, gasp_upc_PTS_t *);
        void *src = (void *) va_arg(argptr, void *);
        size_t n = (int) va_arg(argptr, int);
        TAU_TRACE_GASPI_PUT(dst, n);
        break;
    }
#if 0 /* Disabled until we can prevent double-counting with -optTrackUPCR */
    case GASP_UPC_MEMCPY:
    {
        int src_rank, dst_rank;
        gasp_upc_PTS_t *dst = (gasp_upc_PTS_t *) va_arg(argptr, gasp_upc_PTS_t *);
        gasp_upc_PTS_t *src = (gasp_upc_PTS_t *) va_arg(argptr, gasp_upc_PTS_t *);
        size_t n = (int)va_arg(argptr, int);
        GASPI_UPCALL(context, src_rank=gaspu_upcall_threadof(src));
        GASPI_UPCALL(context, dst_rank=gaspu_upcall_threadof(dst));
        switch(evttype) {
        case GASP_START:
            if (context->mythread == src_rank) {
                TAU_TRACE_SENDMSG(TAU_UPC_TAGID_NEXT, dst_rank, n);
            } else {
                TAU_TRACE_SENDMSG_REMOTE(TAU_UPC_TAGID_NEXT, dst_rank, n, src_rank);
            }
        case GASP_END:
            if (context->mythread == src_rank) {
                TAU_TRACE_RECVMSG_REMOTE(TAU_UPC_TAGID, context->mythread, n, dst_rank);
            } else {
                TAU_TRACE_RECVMSG(TAU_UPC_TAGID, src_rank, n);
            }
        case GASP_ATOMIC:
            TAU_REGISTER_EVENT(upcevent, tagstr);
            TAU_EVENT(upcevent, n);
        }
        break;
    }
#endif
    case GASP_UPC_MEMGET:
    {
        void *dst = (void *) va_arg(argptr, void *);
        gasp_upc_PTS_t *src = (gasp_upc_PTS_t *) va_arg(argptr, gasp_upc_PTS_t *);
        size_t n = (int) va_arg(argptr, int);
        TAU_TRACE_GASPI_GET(src, n);
        break;
    }
    case GASP_UPC_MEMPUT:
    {
        int remote_rank;
        gasp_upc_PTS_t *dst = (gasp_upc_PTS_t*)va_arg(argptr, gasp_upc_PTS_t*);
        void *src = (void*)va_arg(argptr, void*);
        size_t n = (int)va_arg(argptr, int);
        TAU_TRACE_GASPI_PUT(dst, n);
        break;
    }
    case GASP_UPC_MEMSET:
    {
        int remote_rank;
        gasp_upc_PTS_t *dst = (gasp_upc_PTS_t *) va_arg(argptr, gasp_upc_PTS_t *);
        int c = (int) va_arg(argptr, int);
        size_t n = (int) va_arg(argptr, int);
        TAU_TRACE_GASPI_PUT(dst, n);
        break;
    }
#ifdef GASP_UPC_NB_GET_INIT
    case GASP_UPC_NB_GET_INIT:
    {
        int remote_rank;
        gasp_upc_nb_handle_t handle = (gasp_upc_nb_handle_t)NULL;
        int is_relaxed = (int)va_arg(argptr, int);
        void *dst = (void *)va_arg(argptr, void *);
        gasp_upc_PTS_t *src = (gasp_upc_PTS_t *)va_arg(argptr, gasp_upc_PTS_t *);
        size_t n = (size_t)va_arg(argptr, size_t);
        // TODO: Track this correctly
        TAU_REGISTER_EVENT(upcevent, tagstr);
        TAU_EVENT(upcevent, n);
        break;
    }
#endif /* GASP_UPC_NB_GET_INIT */

#ifdef GASP_UPC_NB_PUT_INIT
    case GASP_UPC_NB_PUT_INIT:
    {
        int remote_rank;
        gasp_upc_nb_handle_t handle = GASP_UPC_NB_TRIVIAL;
        int is_relaxed = (int)va_arg(argptr, int);
        gasp_upc_PTS_t *dst = (gasp_upc_PTS_t *)va_arg(argptr, gasp_upc_PTS_t *);
        void *src = (void *)va_arg(argptr, void *);
        size_t n = (size_t)va_arg(argptr, size_t);
        // TODO: Track this correctly
        TAU_REGISTER_EVENT(upcevent, tagstr);
        TAU_EVENT(upcevent, n);
        break;
    }
#endif /* GASP_UPC_NB_PUT_INIT */

    case GASP_UPC_COLLECTIVE_EXIT:
    {
        if (evttype == GASP_END) {
            /* perform graceful tool shutdown */
            TAU_STOP(tagstr);

            /* first, wait for all to arrive */
            GASPI_UPCALL(context, gaspu_barrier());

            context->enabled = 0;

            Tau_stop_top_level_timer_if_necessary();

            GASPI_UPCALL(context, gaspu_barrier());
            /* wait for all threads to finish shutdown */
        }
        break;
    }

    default:
//        fprintf(stderr, "Unknown evttag: %d\n", evttype);
//        fflush(stderr);
        break;
    }

    if(evttype == GASP_END) {
        /* All done, stop the timer */
        TAU_STOP(tagstr);
    }
}

