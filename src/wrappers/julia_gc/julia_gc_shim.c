/*
 * julia_gc_shim.c — Tracks Julia garbage collection.
 *
 * Julia runs the callbacks registered with jl_gc_set_cb_pre_gc / jl_gc_set_cb_post_gc
 * (julia_gcext.h) on the thread that triggered garbage collection to run:
 *   - "pre" once every Julia thread has stopped at a safepoint;
 *   - "post" after the collection and any finalizers. 
 * Callbacks must not call into Julia or allocate Julia objects. 
 *
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define TAU_JULIA_GC_MAX_DEPTH 8

typedef unsigned long tau_group_t;

static int  (*Tau_init_fn)(void);
static void (*Tau_top_level_fn)(void);
static void (*Tau_profile_c_timer_fn)(void **, const char *, const char *, tau_group_t, const char *);
static tau_group_t (*Tau_get_profile_group_fn)(char *);
static void (*Tau_start_timer_fn)(void *, int, int);
static void (*Tau_stop_timer_fn)(void *, int);
static int  (*Tau_get_thread_fn)(void);
static int  (*Tau_lights_out_fn)(void);
static void (*Tau_get_context_userevent_fn)(void **, const char *);
static void (*Tau_context_userevent_fn)(void *, double);

static char gc_group[] = "JULIA_GC";
static void *gc_timer = NULL;
static void *pause_event = NULL;
static void *live_event = NULL;
static int64_t (*live_bytes_fn)(void) = NULL;

static _Thread_local int gc_depth = 0;
static _Thread_local unsigned gc_started = 0;
static _Thread_local double gc_t0[TAU_JULIA_GC_MAX_DEPTH];

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

#define RESOLVE(var, name) \
    if (!(*(void **)(&var) = dlsym(RTLD_DEFAULT, name))) { \
        fprintf(stderr, "tau_julia_gc: %s not found; is libTAU loaded?\n", name); \
        return 1; \
    }

/* Creates the timer and events. `live_bytes` is Julia's jl_gc_live_bytes, or NULL.
 * Returns 0 on success. */
int Tau_julia_gc_init(void *live_bytes)
{
    if (gc_timer == NULL) {
        RESOLVE(Tau_init_fn, "Tau_init_initializeTAU");
        RESOLVE(Tau_top_level_fn, "Tau_create_top_level_timer_if_necessary");
        RESOLVE(Tau_profile_c_timer_fn, "Tau_profile_c_timer");
        RESOLVE(Tau_get_profile_group_fn, "Tau_get_profile_group");
        RESOLVE(Tau_start_timer_fn, "Tau_start_timer");
        RESOLVE(Tau_stop_timer_fn, "Tau_stop_timer");
        RESOLVE(Tau_get_thread_fn, "Tau_get_thread");
        RESOLVE(Tau_lights_out_fn, "Tau_global_getLightsOut");
        RESOLVE(Tau_get_context_userevent_fn, "Tau_get_context_userevent");
        RESOLVE(Tau_context_userevent_fn, "Tau_context_userevent");
        Tau_init_fn();
        Tau_top_level_fn();
        Tau_profile_c_timer_fn(&gc_timer, "Julia GC", "", Tau_get_profile_group_fn(gc_group), gc_group);
        Tau_get_context_userevent_fn(&pause_event, "Julia GC pause (s)");
        Tau_get_context_userevent_fn(&live_event, "Julia GC live bytes after collection");
    }
    live_bytes_fn = (int64_t (*)(void))live_bytes;
    return gc_timer == NULL ? 1 : 0;
}

void Tau_julia_gc_pre(int collection)
{
    int d = gc_depth++;
    if (d >= TAU_JULIA_GC_MAX_DEPTH || gc_timer == NULL || Tau_lights_out_fn()) return;
    gc_started |= 1u << d;
    gc_t0[d] = now_s();
    Tau_start_timer_fn(gc_timer, 0, Tau_get_thread_fn());
}

void Tau_julia_gc_post(int collection)
{
    /* Registered between another thread's pre and post: that post finds depth 0 here. */
    if (gc_depth == 0) return;
    int d = --gc_depth;
    if (d >= TAU_JULIA_GC_MAX_DEPTH || !(gc_started & (1u << d))) return;
    gc_started &= ~(1u << d);
    double pause = now_s() - gc_t0[d];
    Tau_stop_timer_fn(gc_timer, Tau_get_thread_fn());
    /* After the stop, so the events' context is the code that triggered the collection. */
    if (Tau_lights_out_fn()) return;
    Tau_context_userevent_fn(pause_event, pause);
    if (live_bytes_fn) Tau_context_userevent_fn(live_event, (double)live_bytes_fn());
}
