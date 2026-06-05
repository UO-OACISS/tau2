/*
 * blas_server_shim.c — wrapper for OpenBLAS's queue dispatcher.
 *
 * Intercepts and instruments work being dispatched to OpenBLAS
 * worker threads. For use with OpenBLAS as shipped by Julia
 * through tau_julia.`
 *
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct tau_blas_queue {
    void *routine;
    long  position, assigned;
    void *args;
    void *range_m, *range_n, *sa, *sb;
    struct tau_blas_queue *next;
    pthread_mutex_t lock;
    pthread_cond_t  finished;
    int   mode, status;
} tau_blas_queue_t;

#define TAU_BLAS_LEGACY   0x8000U
#define TAU_BLAS_PTHREAD  0x4000U

typedef int  (*blas_kernel_fn_t)(void*, void*, void*, void*, void*, long);
typedef int  (*exec_blas_async_fn_t)(long, tau_blas_queue_t*);
typedef void (*tau_start_fn_t)(const char*);
typedef void (*tau_stop_fn_t)(const char*);

#define TAU_BLAS_SERVER_MAX_SLOTS 64

typedef struct {
    void       * real_fn;   
    const char * name;     
} kernel_slot_t;

static kernel_slot_t   g_slots[TAU_BLAS_SERVER_MAX_SLOTS];
static int             g_slots_used = 0;
static pthread_mutex_t g_slots_lock = PTHREAD_MUTEX_INITIALIZER;
static blas_kernel_fn_t g_thunks[TAU_BLAS_SERVER_MAX_SLOTS];

static exec_blas_async_fn_t real_exec_blas_async = NULL;
static tau_start_fn_t       Tau_start_fn        = NULL;
static tau_stop_fn_t        Tau_stop_fn         = NULL;

static int             g_initialized = 0;
static int             g_disabled    = 0;
static int             g_verbose     = 0;
static pthread_mutex_t g_init_lock   = PTHREAD_MUTEX_INITIALIZER;

static int is_internal_frame_name(const char *name) {
    if (!name) return 1;
    if (strcmp(name, "exec_blas") == 0)         return 1;
    if (strcmp(name, "exec_blas_async") == 0)   return 1;
    if (strcmp(name, "exec_blas_async_wait") == 0) return 1;
    if (strncmp(name, "tau_", 4) == 0)          return 1;
    return 0;
}

// Try to get BLAS routine name from stack
static char * resolve_family_name_from_stack(void) {
    void * frames[12];
    int n = backtrace(frames, 12);
    for (int i = 1; i < n; ++i) {
        Dl_info info;
        if (!dladdr(frames[i], &info) || !info.dli_sname)
            continue;
        if (is_internal_frame_name(info.dli_sname)) 
            continue;
        return strdup(info.dli_sname);
    }
    return NULL;
}

// If can't get name from stack, use offset
static char * fallback_name_for_kernel(void *fn) {
    Dl_info info;
    int have_info = dladdr(fn, &info);
    char buf[96];
    if (have_info && info.dli_fname) {
        const char *base = strrchr(info.dli_fname, '/');
        base = base ? base + 1 : info.dli_fname;
        snprintf(buf, sizeof(buf), "%s+0x%lx", base,
                 (unsigned long)((char*)fn - (char*)info.dli_fbase));
    } else {
        snprintf(buf, sizeof(buf), "blas_kernel@%p", fn);
    }
    return strdup(buf);
}

static int get_or_create_slot(void *fn) {
    // First check if already exists
    pthread_mutex_lock(&g_slots_lock);
    for (int i = 0; i < g_slots_used; ++i) {
        if (g_slots[i].real_fn == fn) {
            pthread_mutex_unlock(&g_slots_lock);
            return i;
        }
    }
    pthread_mutex_unlock(&g_slots_lock);

    // Otherwise, create
    char *family = resolve_family_name_from_stack();

    pthread_mutex_lock(&g_slots_lock);
    for (int i = 0; i < g_slots_used; ++i) {
        if (g_slots[i].real_fn == fn) {
            pthread_mutex_unlock(&g_slots_lock);
            free(family);
            return i;
        }
    }
    if (g_slots_used >= TAU_BLAS_SERVER_MAX_SLOTS) {
        pthread_mutex_unlock(&g_slots_lock);
        free(family);
        if (g_verbose) {
            fprintf(stderr, "tau_blas_server: slot pool exhausted at %d entries\n",
                    TAU_BLAS_SERVER_MAX_SLOTS);
        }
        return -1;
    }
    int idx = g_slots_used++;
    g_slots[idx].real_fn = fn;
    g_slots[idx].name    = family ? family : fallback_name_for_kernel(fn);
    pthread_mutex_unlock(&g_slots_lock);
    if (g_verbose) {
        fprintf(stderr, "tau_blas_server: slot %d -> %s (kernel=%p)\n",
                idx, g_slots[idx].name ? g_slots[idx].name : "?", fn);
    }
    return idx;
}

static int dispatch(int slot, void *args, void *rm, void *rn,
                    void *sa, void *sb, long pos) {
    blas_kernel_fn_t real = (blas_kernel_fn_t)g_slots[slot].real_fn;
    const char      *name = g_slots[slot].name;
    if (Tau_start_fn && name) Tau_start_fn(name);
    int rc = real(args, rm, rn, sa, sb, pos);
    if (Tau_stop_fn && name) Tau_stop_fn(name);
    return rc;
}

#define THUNK_LIST(M) \
    M(0)  M(1)  M(2)  M(3)  M(4)  M(5)  M(6)  M(7)  \
    M(8)  M(9)  M(10) M(11) M(12) M(13) M(14) M(15) \
    M(16) M(17) M(18) M(19) M(20) M(21) M(22) M(23) \
    M(24) M(25) M(26) M(27) M(28) M(29) M(30) M(31) \
    M(32) M(33) M(34) M(35) M(36) M(37) M(38) M(39) \
    M(40) M(41) M(42) M(43) M(44) M(45) M(46) M(47) \
    M(48) M(49) M(50) M(51) M(52) M(53) M(54) M(55) \
    M(56) M(57) M(58) M(59) M(60) M(61) M(62) M(63)

#define MAKE_THUNK(i) \
    static int tau_kernel_thunk_##i(void *args, void *rm, void *rn, \
                                    void *sa, void *sb, long pos) { \
        return dispatch(i, args, rm, rn, sa, sb, pos);              \
    }
THUNK_LIST(MAKE_THUNK)

#define THUNK_ENTRY(i) tau_kernel_thunk_##i,
static void populate_thunk_table(void) {
    blas_kernel_fn_t arr[TAU_BLAS_SERVER_MAX_SLOTS] = { THUNK_LIST(THUNK_ENTRY) };
    memcpy(g_thunks, arr, sizeof(g_thunks));
}

static void ensure_init(void) {
    if (__builtin_expect(g_initialized, 1)) return;
    pthread_mutex_lock(&g_init_lock);
    if (g_initialized) { pthread_mutex_unlock(&g_init_lock); return; }

    const char *opt = getenv("TAU_BLAS_SERVER_HOOK");
    if (opt && opt[0] == '0') g_disabled = 1;
    const char *v = getenv("TAU_BLAS_SERVER_VERBOSE");
    g_verbose = (v && v[0] && v[0] != '0') ? 1 : 0;

    populate_thunk_table();

    dlerror();
    real_exec_blas_async = (exec_blas_async_fn_t)dlsym(RTLD_NEXT, "exec_blas_async");
    if (!real_exec_blas_async) {
        fprintf(stderr, "tau_blas_server: dlsym(RTLD_NEXT, exec_blas_async) failed: %s\n",
                dlerror());
        g_disabled = 1;
    }

    Tau_start_fn = (tau_start_fn_t)dlsym(RTLD_DEFAULT, "Tau_start");
    Tau_stop_fn  = (tau_stop_fn_t)dlsym(RTLD_DEFAULT, "Tau_stop");

    g_initialized = 1;
    pthread_mutex_unlock(&g_init_lock);

    if (g_verbose) {
        fprintf(stderr, "tau_blas_server: initialized (Tau_start=%p, real_exec_blas_async=%p, disabled=%d)\n",
                (void*)Tau_start_fn, (void*)real_exec_blas_async, g_disabled);
    }
}

static inline void maybe_resolve_tau(void) {
    if (!Tau_start_fn) Tau_start_fn = (tau_start_fn_t)dlsym(RTLD_DEFAULT, "Tau_start");
    if (!Tau_stop_fn)  Tau_stop_fn  = (tau_stop_fn_t)dlsym(RTLD_DEFAULT, "Tau_stop");
}

int exec_blas_async(long num, tau_blas_queue_t *queue) {
    ensure_init();
    if (g_disabled || !real_exec_blas_async) {
        return real_exec_blas_async ? real_exec_blas_async(num, queue) : -1;
    }
    maybe_resolve_tau();

    for (tau_blas_queue_t *q = queue; q != NULL; q = q->next) {
        if (q->mode & (TAU_BLAS_LEGACY | TAU_BLAS_PTHREAD)) continue;
        if (!q->routine) continue;
        int slot = get_or_create_slot(q->routine);
        if (slot >= 0) q->routine = (void*)g_thunks[slot];
    }
    return real_exec_blas_async(num, queue);
}
