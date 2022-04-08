#include <Profile/Profiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <dlfcn.h>

static const char * tau_orig_libname = "libchpl.so";
static void *tau_handle = NULL;


#ifndef TAU_GROUP_TAU_CHPL
#define TAU_GROUP_TAU_CHPL TAU_GET_PROFILE_GROUP("TAU_CHPL")
#endif /* TAU_GROUP_TAU_CHPL */ 

typedef enum {
    memory_order_relaxed,
    memory_order_consume,
    memory_order_acquire,
    memory_order_release,
    memory_order_acq_rel,
    memory_order_seq_cst
} memory_order;

typedef int32_t c_nodeid_t;
typedef int32_t chpl_bool32;

#define WRAP_CHPL_COMM_ATOMIC_WRITE(type)                               \
  extern void chpl_comm_atomic_write_ ## type                           \
         (void* a1, c_nodeid_t a2, void* a3,                 \
          memory_order a4, int a5, int32_t a6) {                     \
                                                                        \
        static void (*chpl_comm_atomic_write_h) (void * a1, c_nodeid_t a2, void * a3, memory_order a4, int a5, int32_t a6) = NULL; \
        TAU_PROFILE_TIMER(t,"void chpl_comm_atomic_write_" #type "(void *, c_nodeit_t, void *, memory_order, int, int32_t) C", "", TAU_GROUP_TAU_CHPL); \
        if (tau_handle == NULL) \
            tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); \
                                                                      \
        if (tau_handle == NULL) { \
            perror("Error opening library in dlopen call"); \
            fprintf(stderr, "%s\n", dlerror()); \
            return;\
        } else { \
            if (chpl_comm_atomic_write_h == NULL)\
               chpl_comm_atomic_write_h = dlsym(tau_handle,"chpl_comm_atomic_write_" #type); \
            if (chpl_comm_atomic_write_h == NULL) {\
               perror("Error obtaining symbol info from dlopen'ed lib"); \
               fprintf(stderr, "%s\n", dlerror()); \
            return;\
            }\
        }\
        TAU_PROFILE_START(t);\
        (*chpl_comm_atomic_write_h) ( a1,  a2,  a3,  a4, a5, a6);\
        TAU_PROFILE_STOP(t);\
        return;\
    }


WRAP_CHPL_COMM_ATOMIC_WRITE(int32)
WRAP_CHPL_COMM_ATOMIC_WRITE(int64)
WRAP_CHPL_COMM_ATOMIC_WRITE(uint32)
WRAP_CHPL_COMM_ATOMIC_WRITE(uint64)
WRAP_CHPL_COMM_ATOMIC_WRITE(real32)
WRAP_CHPL_COMM_ATOMIC_WRITE(real64)


#define WRAP_CHPL_COMM_ATOMIC_READ(type)                               \
  extern void chpl_comm_atomic_read_ ## type                           \
         (void* a1, c_nodeid_t a2, void* a3,                 \
          memory_order a4, int a5, int32_t a6) {                     \
                                                                        \
        static void (*chpl_comm_atomic_read_h) (void * a1, c_nodeid_t a2, void * a3, memory_order a4, int a5, int32_t a6) = NULL; \
        TAU_PROFILE_TIMER(t,"void chpl_comm_atomic_read_" #type "(void *, c_nodeit_t, void *, memory_order, int, int32_t) C", "", TAU_GROUP_TAU_CHPL); \
        if (tau_handle == NULL) \
            tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); \
                                                                      \
        if (tau_handle == NULL) { \
            perror("Error opening library in dlopen call"); \
            fprintf(stderr, "%s\n", dlerror()); \
            return;\
        } else { \
            if (chpl_comm_atomic_read_h == NULL)\
               chpl_comm_atomic_read_h = dlsym(tau_handle,"chpl_comm_atomic_read_" #type); \
            if (chpl_comm_atomic_read_h == NULL) {\
               perror("Error obtaining symbol info from dlopen'ed lib"); \
            fprintf(stderr, "%s\n", dlerror()); \
            return;\
            }\
        }\
        TAU_PROFILE_START(t);\
        (*chpl_comm_atomic_read_h) ( a1,  a2,  a3,  a4, a5, a6);\
        TAU_PROFILE_STOP(t);\
        return;\
    }

WRAP_CHPL_COMM_ATOMIC_READ(int32)
WRAP_CHPL_COMM_ATOMIC_READ(int64)
WRAP_CHPL_COMM_ATOMIC_READ(uint32)
WRAP_CHPL_COMM_ATOMIC_READ(uint64)
WRAP_CHPL_COMM_ATOMIC_READ(real32)
WRAP_CHPL_COMM_ATOMIC_READ(real64)

#define WRAP_CHPL_COMM_ATOMIC_XCHG(type)                               \
  extern void chpl_comm_atomic_xchg_ ## type                           \
         (void* a1, c_nodeid_t a2, void* a3, void* a4,                 \
          memory_order a5, int a6, int32_t a7) {                     \
                                                                        \
        static void (*chpl_comm_atomic_xchg_h) (void * a1, c_nodeid_t a2, void * a3, void * a4, memory_order a5, int a6, int32_t a7) = NULL; \
        TAU_PROFILE_TIMER(t,"void chpl_comm_atomic_xchg_" #type "(void *, c_nodeit_t, void *, void *, memory_order, int, int32_t) C", "", TAU_GROUP_TAU_CHPL); \
        if (tau_handle == NULL) \
            tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); \
                                                                      \
        if (tau_handle == NULL) { \
            perror("Error opening library in dlopen call"); \
            fprintf(stderr, "%s\n", dlerror()); \
            return;\
        } else { \
            if (chpl_comm_atomic_xchg_h == NULL)\
               chpl_comm_atomic_xchg_h = dlsym(tau_handle,"chpl_comm_atomic_xchg_" #type); \
            if (chpl_comm_atomic_xchg_h == NULL) {\
               perror("Error obtaining symbol info from dlopen'ed lib"); \
            fprintf(stderr, "%s\n", dlerror()); \
            return;\
            }\
        }\
        TAU_PROFILE_START(t);\
        (*chpl_comm_atomic_xchg_h) ( a1,  a2,  a3,  a4, a5, a6, a7);\
        TAU_PROFILE_STOP(t);\
        return;\
    }

WRAP_CHPL_COMM_ATOMIC_XCHG(int32)
WRAP_CHPL_COMM_ATOMIC_XCHG(int64)
WRAP_CHPL_COMM_ATOMIC_XCHG(uint32)
WRAP_CHPL_COMM_ATOMIC_XCHG(uint64)
WRAP_CHPL_COMM_ATOMIC_XCHG(real32)
WRAP_CHPL_COMM_ATOMIC_XCHG(real64)

#define WRAP_CHPL_COMM_ATOMIC_CMPXCHG(type)                               \
  extern void chpl_comm_atomic_cmpxchg_ ## type                           \
         (void* a1, void * a2, c_nodeid_t a3, void* a4,                 \
          chpl_bool32 * a5, memory_order a6, memory_order a7, int a8, int32_t a9) {                     \
                                                                        \
        static void (*chpl_comm_atomic_cmpxchg_h) (void * a1, void * a2, c_nodeid_t a3, void * a4, chpl_bool32 * a5, memory_order a6, memory_order a7, int a8, int32_t a9) = NULL; \
        TAU_PROFILE_TIMER(t,"void chpl_comm_atomic_cmpxchg_" #type "(void *, void *, c_nodeid_t, void *, chpl_bool32 *, memory_order, memory_order, int, int32_t) C", "", TAU_GROUP_TAU_CHPL); \
        if (tau_handle == NULL) \
            tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); \
                                                                      \
        if (tau_handle == NULL) { \
            perror("Error opening library in dlopen call"); \
            fprintf(stderr, "%s\n", dlerror()); \
            return;\
        } else { \
            if (chpl_comm_atomic_cmpxchg_h == NULL)\
               chpl_comm_atomic_cmpxchg_h = dlsym(tau_handle,"chpl_comm_atomic_cmpxchg_" #type); \
            if (chpl_comm_atomic_cmpxchg_h == NULL) {\
               perror("Error obtaining symbol info from dlopen'ed lib"); \
            fprintf(stderr, "%s\n", dlerror()); \
            return;\
            }\
        }\
        TAU_PROFILE_START(t);\
        (*chpl_comm_atomic_cmpxchg_h) ( a1,  a2,  a3,  a4, a5, a6, a7, a8, a9);\
        TAU_PROFILE_STOP(t);\
        return;\
    }

WRAP_CHPL_COMM_ATOMIC_CMPXCHG(int32)
WRAP_CHPL_COMM_ATOMIC_CMPXCHG(int64)
WRAP_CHPL_COMM_ATOMIC_CMPXCHG(uint32)
WRAP_CHPL_COMM_ATOMIC_CMPXCHG(uint64)
WRAP_CHPL_COMM_ATOMIC_CMPXCHG(real32)
WRAP_CHPL_COMM_ATOMIC_CMPXCHG(real64)

#define WRAP_CHPL_COMM_ATOMIC_NONFETCH_BINARY(op, type)                 \
  void chpl_comm_atomic_ ## op ## _ ## type                             \
         (void* a1, c_nodeid_t a2, void* a3,                 \
          memory_order a4, int a5, int32_t a6) { \
                                                 \
            static void (*chpl_comm_atomic_nonfetch_binary_h) (void * a1, c_nodeid_t a2, void * a3, memory_order a4, int a5, int32_t a6) = NULL; \
            TAU_PROFILE_TIMER(t,"void chpl_comm_atomic_" #op "_" #type "(void *, c_nodeit_t, void *, memory_order, int, int32_t) C", "", TAU_GROUP_TAU_CHPL); \
            if (tau_handle == NULL) \
                tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); \
                                                                        \
            if (tau_handle == NULL) { \
                perror("Error opening library in dlopen call"); \
            fprintf(stderr, "%s\n", dlerror()); \
                return;\
            } else { \
                if (chpl_comm_atomic_nonfetch_binary_h == NULL)\
                chpl_comm_atomic_nonfetch_binary_h = dlsym(tau_handle,"chpl_comm_atomic_" #op "_" #type); \
                if (chpl_comm_atomic_nonfetch_binary_h == NULL) {\
                perror("Error obtaining symbol info from dlopen'ed lib"); \
            fprintf(stderr, "%s\n", dlerror()); \
                return;\
                }\
            }\
            TAU_PROFILE_START(t);\
            (*chpl_comm_atomic_nonfetch_binary_h) ( a1,  a2,  a3,  a4, a5, a6);\
            TAU_PROFILE_STOP(t);\
            return;\
      }

                                                                       
#define WRAP_CHPL_COMM_ATOMIC_NONFETCH_UNORDERED_BINARY(op, type)       \
  void chpl_comm_atomic_ ## op ## _unordered_ ## type                   \
         (void* a1, c_nodeid_t a2, void* a3,                 \
          int a4, int32_t a5) { \
            static void (*chpl_comm_atomic_nonfetch_unordered_binary_h) (void * a1, c_nodeid_t a2, void * a3, int a4, int32_t a5) = NULL; \
            TAU_PROFILE_TIMER(t,"void chpl_comm_atomic_" #op "_unordered_" #type "(void *, c_nodeit_t, void *, int, int32_t) C", "", TAU_GROUP_TAU_CHPL); \
            if (tau_handle == NULL) \
                tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); \
                                                                        \
            if (tau_handle == NULL) { \
                perror("Error opening library in dlopen call"); \
            fprintf(stderr, "%s\n", dlerror()); \
                return;\
            } else { \
                if (chpl_comm_atomic_nonfetch_unordered_binary_h == NULL)\
                chpl_comm_atomic_nonfetch_unordered_binary_h = dlsym(tau_handle,"chpl_comm_atomic_" #op "_unordered_" #type); \
                if (chpl_comm_atomic_nonfetch_unordered_binary_h == NULL) {\
                perror("Error obtaining symbol info from dlopen'ed lib"); \
            fprintf(stderr, "%s\n", dlerror()); \
                return;\
                }\
            }\
            TAU_PROFILE_START(t);\
            (*chpl_comm_atomic_nonfetch_unordered_binary_h) ( a1,  a2,  a3,  a4, a5);\
            TAU_PROFILE_STOP(t);\
            return;\
      }

#define WRAP_CHPL_COMM_ATOMIC_FETCH_BINARY(op, type)                    \
  void chpl_comm_atomic_fetch_ ## op ## _ ## type                       \
         (void* a1, c_nodeid_t a2, void* a3, void* a4,   \
          memory_order a5, int a6, int32_t a7) {                     \
            static void (*chpl_comm_atomic_fetch_binary_h) (void * a1, c_nodeid_t a2, void * a3, void * a4, memory_order a5, int a6, int32_t a7) = NULL; \
            TAU_PROFILE_TIMER(t,"void chpl_comm_atomic_fetch" #op "_" #type "(void *, c_nodeit_t, void *, void *, memory_order, int, int32_t) C", "", TAU_GROUP_TAU_CHPL); \
            if (tau_handle == NULL) \
                tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); \
                                                                        \
            if (tau_handle == NULL) { \
                perror("Error opening library in dlopen call"); \
            fprintf(stderr, "%s\n", dlerror()); \
                return;\
            } else { \
                if (chpl_comm_atomic_fetch_binary_h == NULL)\
                chpl_comm_atomic_fetch_binary_h = dlsym(tau_handle,"chpl_comm_atomic_fetch_" #op "_" #type); \
                if (chpl_comm_atomic_fetch_binary_h == NULL) {\
                perror("Error obtaining symbol info from dlopen'ed lib"); \
            fprintf(stderr, "%s\n", dlerror()); \
                return;\
                }\
            }\
            TAU_PROFILE_START(t);\
            (*chpl_comm_atomic_fetch_binary_h) ( a1,  a2,  a3,  a4, a5, a6, a7);\
            TAU_PROFILE_STOP(t);\
            return;\
      }

#define WRAP_CHPL_COMM_ATOMIC_BINARY(op, type)                          \
  WRAP_CHPL_COMM_ATOMIC_NONFETCH_BINARY(op, type)                       \
  WRAP_CHPL_COMM_ATOMIC_NONFETCH_UNORDERED_BINARY(op, type)             \
  WRAP_CHPL_COMM_ATOMIC_FETCH_BINARY(op, type)


WRAP_CHPL_COMM_ATOMIC_BINARY(and, int32)
WRAP_CHPL_COMM_ATOMIC_BINARY(and, int64)
WRAP_CHPL_COMM_ATOMIC_BINARY(and, uint32)
WRAP_CHPL_COMM_ATOMIC_BINARY(and, uint64)

WRAP_CHPL_COMM_ATOMIC_BINARY(or, int32)
WRAP_CHPL_COMM_ATOMIC_BINARY(or, int64)
WRAP_CHPL_COMM_ATOMIC_BINARY(or, uint32)
WRAP_CHPL_COMM_ATOMIC_BINARY(or, uint64)

WRAP_CHPL_COMM_ATOMIC_BINARY(xor, int32)
WRAP_CHPL_COMM_ATOMIC_BINARY(xor, int64)
WRAP_CHPL_COMM_ATOMIC_BINARY(xor, uint32)
WRAP_CHPL_COMM_ATOMIC_BINARY(xor, uint64)

WRAP_CHPL_COMM_ATOMIC_BINARY(add, int32)
WRAP_CHPL_COMM_ATOMIC_BINARY(add, int64)
WRAP_CHPL_COMM_ATOMIC_BINARY(add, uint32)
WRAP_CHPL_COMM_ATOMIC_BINARY(add, uint64)
WRAP_CHPL_COMM_ATOMIC_BINARY(add, real32)
WRAP_CHPL_COMM_ATOMIC_BINARY(add, real64)

WRAP_CHPL_COMM_ATOMIC_BINARY(sub, int32)
WRAP_CHPL_COMM_ATOMIC_BINARY(sub, int64)
WRAP_CHPL_COMM_ATOMIC_BINARY(sub, uint32)
WRAP_CHPL_COMM_ATOMIC_BINARY(sub, uint64)
WRAP_CHPL_COMM_ATOMIC_BINARY(sub, real32)
WRAP_CHPL_COMM_ATOMIC_BINARY(sub, real64)


void chpl_comm_atomic_unordered_task_fence(void) {
  static void (*chpl_comm_atomic_unordered_task_fence_h) (void) = NULL;
  TAU_PROFILE_TIMER(t,"void chpl_comm_atomic_unordered_task_fence(void) C", "", TAU_GROUP_TAU_CHPL);
  if (tau_handle == NULL) 
    tau_handle = (void *) dlopen(tau_orig_libname, RTLD_NOW); 

  if (tau_handle == NULL) { 
    perror("Error opening library in dlopen call"); 
            fprintf(stderr, "%s\n", dlerror()); \
    return;
  } else { 
    if (chpl_comm_atomic_unordered_task_fence_h == NULL)
      chpl_comm_atomic_unordered_task_fence_h = dlsym(tau_handle,"chpl_comm_atomic_unordered_task_fence"); 
    if (chpl_comm_atomic_unordered_task_fence_h == NULL) {
      perror("Error obtaining symbol info from dlopen'ed lib"); 
            fprintf(stderr, "%s\n", dlerror()); \
      return;
    }
  }
  TAU_PROFILE_START(t);
  (*chpl_comm_atomic_unordered_task_fence_h) ();
  TAU_PROFILE_STOP(t);
  return;

}


