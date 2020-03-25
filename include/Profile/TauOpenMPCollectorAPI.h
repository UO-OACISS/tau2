#ifndef TAU_OPENMP_COLLECTOR_API_H
#define TAU_OPENMP_COLLECTOR_API_H

/* An array of this struct is shared by all threads. To make sure we don't have false
 * sharing, the struct is 64 bytes in size, so that it fits exactly in
 * one (or two) cache lines. That way, when one thread updates its data
 * in the array, it won't invalidate the cache line for other threads. 
 * This is very important with timers, as all threads are entering timers
 * at the same time, and every thread will invalidate the cache line
 * otherwise. */
struct Tau_collector_status_flags {
    int idle; // 4 bytes
    int busy; // 4 bytes
    int parallel; // 4 bytes
    int ordered_region_wait; // 4 bytes
    int ordered_region; // 4 bytes
    int task_exec; // 4 bytes
    char *timerContext; // 8 bytes(?)
    char *activeTimerContext; // 8 bytes(?)
    void *signal_message; // preallocated message for signal handling, 8 bytes
    char _pad[64-((sizeof(void*))+(2*sizeof(char*))+(6*sizeof(int)))];
};

static char* __UNKNOWN__ = "UNKNOWN";

extern struct Tau_collector_status_flags Tau_collector_flags[TAU_MAX_THREADS];

#ifdef __cplusplus
#define TAU_CDECL "C"
#else
#define TAU_CDECL 
#endif

// These are functions we will reuse in the TauOMPT.cpp file
extern TAU_CDECL void Tau_get_current_region_context(int tid);
extern TAU_CDECL void Tau_omp_start_timer(const char * state, int tid, int use_context);
extern TAU_CDECL void Tau_omp_stop_timer(const char * state, int tid, int use_context);
extern TAU_CDECL void Tau_profile_exit_most_threads();

#undef TAU_CDECL

#endif //TAU_OPENMP_COLLECTOR_API_H
