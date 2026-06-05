/*
 * lbt_shim.c — TAU wrapper that installs Tau_start/Tau_stop wrappers around
 * BLAS Level-3 routines via libblastrampoline's lbt_set_forward() API.
 * For use with tau_julia.
 *
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LBT_INTERFACE_LP64              32
#define LBT_INTERFACE_ILP64             64
#define LBT_COMPLEX_RETSTYLE_NORMAL      0
#define LBT_F2C_PLAIN                    0

typedef int32_t (*lbt_set_forward_t)(const char*, const void*, int32_t, int32_t, int32_t, int32_t);
typedef const void* (*lbt_get_forward_t)(const char*, int32_t, int32_t);

typedef void (*tau_start_t)(const char*);
typedef void (*tau_stop_t)(const char*);

static lbt_set_forward_t  lbt_set_forward_fn  = NULL;
static lbt_get_forward_t  lbt_get_forward_fn  = NULL;
static tau_start_t        Tau_start_fn        = NULL;
static tau_stop_t         Tau_stop_fn         = NULL;

static int tau_lbt_verbose = 0;

#define ARGS_10   void* a1, void* a2, void* a3, void* a4, void* a5, void* a6, void* a7, void* a8, void* a9, void* a10
#define CALL_10   a1, a2, a3, a4, a5, a6, a7, a8, a9, a10
#define ARGS_11   ARGS_10, void* a11
#define CALL_11   CALL_10, a11
#define ARGS_12   ARGS_11, void* a12
#define CALL_12   CALL_11, a12
#define ARGS_13   ARGS_12, void* a13
#define CALL_13   CALL_12, a13

typedef void (*blas_fn10_t)(ARGS_10);
typedef void (*blas_fn11_t)(ARGS_11);
typedef void (*blas_fn12_t)(ARGS_12);
typedef void (*blas_fn13_t)(ARGS_13);

#define MAKE_SHIM(name, fortran_name, arity)                                   \
    static blas_fn##arity##_t real_##name##_lp64;                              \
    static blas_fn##arity##_t real_##name##_ilp64;                             \
    static void tau_##name##_lp64(ARGS_##arity) {                              \
        Tau_start_fn(fortran_name);                                            \
        real_##name##_lp64(CALL_##arity);                                      \
        Tau_stop_fn(fortran_name);                                             \
    }                                                                          \
    static void tau_##name##_ilp64(ARGS_##arity) {                             \
        Tau_start_fn(fortran_name);                                            \
        real_##name##_ilp64(CALL_##arity);                                     \
        Tau_stop_fn(fortran_name);                                             \
    }

MAKE_SHIM(sgemm,  "sgemm_",  13)
MAKE_SHIM(dgemm,  "dgemm_",  13)
MAKE_SHIM(cgemm,  "cgemm_",  13)
MAKE_SHIM(zgemm,  "zgemm_",  13)

MAKE_SHIM(ssymm,  "ssymm_",  12)
MAKE_SHIM(dsymm,  "dsymm_",  12)
MAKE_SHIM(csymm,  "csymm_",  12)
MAKE_SHIM(zsymm,  "zsymm_",  12)

MAKE_SHIM(ssyrk,  "ssyrk_",  10)
MAKE_SHIM(dsyrk,  "dsyrk_",  10)
MAKE_SHIM(csyrk,  "csyrk_",  10)
MAKE_SHIM(zsyrk,  "zsyrk_",  10)

MAKE_SHIM(ssyr2k, "ssyr2k_", 12)
MAKE_SHIM(dsyr2k, "dsyr2k_", 12)
MAKE_SHIM(csyr2k, "csyr2k_", 12)
MAKE_SHIM(zsyr2k, "zsyr2k_", 12)

MAKE_SHIM(strmm,  "strmm_",  11)
MAKE_SHIM(dtrmm,  "dtrmm_",  11)
MAKE_SHIM(ctrmm,  "ctrmm_",  11)
MAKE_SHIM(ztrmm,  "ztrmm_",  11)

MAKE_SHIM(strsm,  "strsm_",  11)
MAKE_SHIM(dtrsm,  "dtrsm_",  11)
MAKE_SHIM(ctrsm,  "ctrsm_",  11)
MAKE_SHIM(ztrsm,  "ztrsm_",  11)

MAKE_SHIM(chemm,  "chemm_",  12)
MAKE_SHIM(zhemm,  "zhemm_",  12)

MAKE_SHIM(cherk,  "cherk_",  10)
MAKE_SHIM(zherk,  "zherk_",  10)

MAKE_SHIM(cher2k, "cher2k_", 12)
MAKE_SHIM(zher2k, "zher2k_", 12)

typedef struct {
    const char* lbt_name;
    void*       shim_lp64;
    void*       shim_ilp64;
    void**      real_lp64_slot;
    void**      real_ilp64_slot;
} blas_shim_entry_t;

#define ENTRY(name) {                                       \
    #name "_",                                              \
    (void*)tau_##name##_lp64,                               \
    (void*)tau_##name##_ilp64,                              \
    (void**)&real_##name##_lp64,                            \
    (void**)&real_##name##_ilp64,                           \
}

static const blas_shim_entry_t SHIM_TABLE[] = {
    ENTRY(sgemm),  ENTRY(dgemm),  ENTRY(cgemm),  ENTRY(zgemm),
    ENTRY(ssymm),  ENTRY(dsymm),  ENTRY(csymm),  ENTRY(zsymm),
    ENTRY(ssyrk),  ENTRY(dsyrk),  ENTRY(csyrk),  ENTRY(zsyrk),
    ENTRY(ssyr2k), ENTRY(dsyr2k), ENTRY(csyr2k), ENTRY(zsyr2k),
    ENTRY(strmm),  ENTRY(dtrmm),  ENTRY(ctrmm),  ENTRY(ztrmm),
    ENTRY(strsm),  ENTRY(dtrsm),  ENTRY(ctrsm),  ENTRY(ztrsm),
    ENTRY(chemm),  ENTRY(zhemm),
    ENTRY(cherk),  ENTRY(zherk),
    ENTRY(cher2k), ENTRY(zher2k),
};
static const size_t SHIM_COUNT = sizeof(SHIM_TABLE) / sizeof(SHIM_TABLE[0]);

// Reload libblastrampoline to force it into global scope.
// (Julia loads it without RTLD_GLOBAL)
static void* find_loaded_lib_handle(const char* name_substring) {
    FILE* maps = fopen("/proc/self/maps", "r");
    if (!maps) return NULL;
    char line[4096];
    char path[4096] = {0};
    while (fgets(line, sizeof(line), maps)) {
        const char* slash = strchr(line, '/');
        if (!slash) continue;
        size_t len = strlen(slash);
        while (len > 0 && (slash[len-1] == '\n' || slash[len-1] == ' ')) len--;
        if (len == 0 || len >= sizeof(path)) continue;
        memcpy(path, slash, len);
        path[len] = '\0';
        if (strstr(path, name_substring)) {
            fclose(maps);
            return dlopen(path, RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
        }
    }
    fclose(maps);
    return NULL;
}

static void* resolve_sym(const char* sym, const char* lib_substring) {
    void* p = dlsym(RTLD_DEFAULT, sym);
    if (p) return p;
    void* handle = find_loaded_lib_handle(lib_substring);
    if (!handle) return NULL;
    return dlsym(handle, sym);
}

static int resolve_required_symbols(void) {
    lbt_set_forward_fn = (lbt_set_forward_t)resolve_sym("lbt_set_forward", "libblastrampoline");
    lbt_get_forward_fn = (lbt_get_forward_t)resolve_sym("lbt_get_forward", "libblastrampoline");
    Tau_start_fn       = (tau_start_t)resolve_sym("Tau_start", "libTAU");
    Tau_stop_fn        = (tau_stop_t)resolve_sym("Tau_stop",  "libTAU");

    if (!lbt_set_forward_fn || !lbt_get_forward_fn) {
        fprintf(stderr, "tau_lbt_shim: could not resolve libblastrampoline symbols; is it loaded?\n");
        return -1;
    }
    if (!Tau_start_fn || !Tau_stop_fn) {
        fprintf(stderr, "tau_lbt_shim: could not resolve Tau_start/Tau_stop; is libTAU loaded?\n");
        return -1;
    }
    return 0;
}

int tau_lbt_install(void) {
    const char* env = getenv("TAU_BLAS_HOOK_VERBOSE");
    tau_lbt_verbose = (env && env[0] && env[0] != '0') ? 1 : 0;

    if (resolve_required_symbols() != 0) {
        return -1;
    }

    int installed = 0;
    for (size_t i = 0; i < SHIM_COUNT; ++i) {
        const blas_shim_entry_t* e = &SHIM_TABLE[i];

        const void* real_lp64 = lbt_get_forward_fn(e->lbt_name, LBT_INTERFACE_LP64, LBT_F2C_PLAIN);
        if (real_lp64) {
            *e->real_lp64_slot = (void*)real_lp64;
            int rc = lbt_set_forward_fn(e->lbt_name, e->shim_lp64,
                                        LBT_INTERFACE_LP64,
                                        LBT_COMPLEX_RETSTYLE_NORMAL,
                                        LBT_F2C_PLAIN, tau_lbt_verbose);
            if (rc == 0) installed++;
            else if (tau_lbt_verbose) fprintf(stderr, "tau_lbt_shim: lbt_set_forward(%s, LP64) -> %d\n", e->lbt_name, rc);
        }

        const void* real_ilp64 = lbt_get_forward_fn(e->lbt_name, LBT_INTERFACE_ILP64, LBT_F2C_PLAIN);
        if (real_ilp64) {
            *e->real_ilp64_slot = (void*)real_ilp64;
            int rc = lbt_set_forward_fn(e->lbt_name, e->shim_ilp64,
                                        LBT_INTERFACE_ILP64,
                                        LBT_COMPLEX_RETSTYLE_NORMAL,
                                        LBT_F2C_PLAIN, tau_lbt_verbose);
            if (rc == 0) installed++;
            else if (tau_lbt_verbose) fprintf(stderr, "tau_lbt_shim: lbt_set_forward(%s, ILP64) -> %d\n", e->lbt_name, rc);
        }
    }

    if (tau_lbt_verbose) {
        fprintf(stderr, "tau_lbt_shim: installed %d BLAS interception slots\n", installed);
    }
    return installed;
}

int tau_lbt_uninstall(void) {
    if (!lbt_set_forward_fn) return 0;
    int restored = 0;
    for (size_t i = 0; i < SHIM_COUNT; ++i) {
        const blas_shim_entry_t* e = &SHIM_TABLE[i];
        if (*e->real_lp64_slot) {
            lbt_set_forward_fn(e->lbt_name, *e->real_lp64_slot,
                               LBT_INTERFACE_LP64,
                               LBT_COMPLEX_RETSTYLE_NORMAL,
                               LBT_F2C_PLAIN, 0);
            restored++;
        }
        if (*e->real_ilp64_slot) {
            lbt_set_forward_fn(e->lbt_name, *e->real_ilp64_slot,
                               LBT_INTERFACE_ILP64,
                               LBT_COMPLEX_RETSTYLE_NORMAL,
                               LBT_F2C_PLAIN, 0);
            restored++;
        }
    }
    return restored;
}
