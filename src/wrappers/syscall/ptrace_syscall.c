// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define _GNU_SOURCE
#include <TAU.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

// For the powerpc implementation, see glibc source code at glibc/sysdeps/unix/sysv/linux/powerpc/libc-start.c
#if defined(__powerpc64__) || defined(__PPC64__) || defined(__APPLE__)
#include <link.h>
#endif

#include <tracee.h>

#if defined(__GNUC__)
#define __TAU_FUNCTION__ __PRETTY_FUNCTION__
#else
#define __TAU_FUNCTION__ __func__
#endif

// #define DEBUG_PTRACE

#ifdef DEBUG_PTRACE
#define DEBUG_PRINT(...)                                                                                               \
    fprintf(stderr, __VA_ARGS__);                                                                                      \
    fflush(stderr);
#else
#define DEBUG_PRINT(...)
#endif

static pid_t rpid;

extern void Tau_profile_exit_all_threads(void);
extern int Tau_init_initializeTAU(void);
extern char* TauEnv_get_tracedir(void);
extern void TauEnv_set_tracedir(const char *);
extern void TauEnv_set_tracedir(const char *);
extern int TauEnv_get_tracing(void);

void __attribute__((constructor)) taupreload_init(void);

void taupreload_init(void)
{
    shared_num_tasks =
        (volatile int *)mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    waiting_for_ack =
        (volatile int *)mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    parent_has_dumped =
        (volatile int *)mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    task_creator_thread_tid =
        (volatile int *)mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    *shared_num_tasks = 0;
    *waiting_for_ack = 0;
    *parent_has_dumped = 0;
    *task_creator_thread_tid = -1;

    waiting_for_ack_mutex = (pthread_mutex_t *)mmap(NULL, sizeof(pthread_mutex_t), PROT_READ | PROT_WRITE,
                                                    MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    waiting_for_ack_cond =
        (pthread_cond_t *)mmap(NULL, sizeof(pthread_cond_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    pthread_mutexattr_t mutex_attr;
    pthread_condattr_t cond_attr;

    pthread_mutexattr_init(&mutex_attr);
    pthread_condattr_init(&cond_attr);

    pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
    pthread_condattr_setpshared(&cond_attr, PTHREAD_PROCESS_SHARED);

    pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_ERRORCHECK);

    pthread_mutex_init(waiting_for_ack_mutex, &mutex_attr);
    pthread_cond_init(waiting_for_ack_cond, &cond_attr);

    rpid = fork();
}

#if defined(__powerpc64__) || defined(__PPC64__) || defined(__APPLE__)
static int (*main_real)(int, char **, char **, void *);

int taupreload_main(int argc, char **argv, char **envp, void *other)
#else
static int (*main_real)(int, char **, char **);

int taupreload_main(int argc, char **argv, char **envp)
#endif
{
    // prevent re-entry
    static int _reentry = 0;
    if (_reentry > 0)
        return -1;
    _reentry = 1;

    int ret;

    // does little, but does something
    TAU_INIT(&argc, &argv);
    // apparently is the real initialization.
    Tau_init_initializeTAU();
    Tau_create_top_level_timer_if_necessary();

    int tmp = TAU_PROFILE_GET_NODE();
    if (tmp == -1)
    {
        TAU_PROFILE_SET_NODE(0);
    }

    if (rpid == 0)
    {
        /* Child */
        pid_t ppid = getppid();

        if (TauEnv_get_tracing())
        {
            // Here we have a trouble: the events.edf file of the child and the parent are different.
            // The parent's file contains only the syscalls's metadata; the child's file contains the metadata of everything else.
            // It may be difficult to merge the two events.edf files, so for the moment, we separate them.
            // This way, it's still possible to use tools like tau_merge to merge the traces files when using the tau format
            // (To use tau_merge, we have to specify the correct events file for each tautrace to be associated with.) 
            const char* tautrace_dir_name = TauEnv_get_tracedir();
            const int buffer_size = 4096;
            char buf[buffer_size];
            // The tautrace directory will contains the events.edf files and the tautraces of the child
            printf("WARNING: TAU_TRACE enabled. The syscalls tautrace files will be in %s/%s/%s directory\n", tautrace_dir_name, "tautrace", "syscall");
            printf("         This folder may contain useless and empty tautrace.x.y.z.trc (like tautrace.x.0.0.trc) which have the same name of files in %s/%s.\n", tautrace_dir_name, "tautrace");
            printf("         The correct ones are always the ones in %s/%s\n", tautrace_dir_name, "tautrace");
            fflush(stdout);
            // Create the folders
            snprintf(buf, buffer_size, "%s/%s", tautrace_dir_name, "tautrace");
            mkdir(buf, 0700);
            TauEnv_set_tracedir(buf);
            // For the parent
            snprintf(buf, buffer_size, "%s/%s/%s", tautrace_dir_name, "tautrace", "syscall");
            mkdir(buf, 0700);
            fflush(stdout);
        }

        void *handle;
        TAU_PROFILER_CREATE(handle, __TAU_FUNCTION__, "", TAU_DEFAULT);
        TAU_PROFILER_START(handle);

        prepare_to_be_tracked(ppid);

#if defined(__powerpc64__) || defined(__PPC64__) || defined(__APPLE__)
        ret = main_real(argc, argv, envp, other);
#else
        ret = main_real(argc, argv, envp);
#endif

        kill(ppid, SIG_STOP_PTRACE);
        DEBUG_PRINT("%d just sent signal SIG_STOP_PTRACE to %d\n", getpid(), ppid);

        while (!(*parent_has_dumped))
        {
            // Lazy solution to a minor problem
            usleep(100);
        }

        pthread_join(task_creator_thread, NULL);

        TAU_PROFILER_STOP(handle);
    }
    else
    {
        /* Parent */


        if (TauEnv_get_tracing())
        {
            const char* tautrace_dir_name = TauEnv_get_tracedir();
            const int buffer_size = 4096;
            char buf[buffer_size];
            snprintf(buf, buffer_size, "%s/%s", tautrace_dir_name, "tautrace/syscall");
            // The child creates the directory
            TauEnv_set_tracedir(buf);
        }

        ret = track_process(rpid);
        DEBUG_PRINT("track_process done with ret = %d\n", ret);

        munmap((int *)shared_num_tasks, sizeof(int));
        munmap((int *)waiting_for_ack, sizeof(int));
        munmap((int *)parent_has_dumped, sizeof(int));
        munmap((int *)task_creator_thread_tid, sizeof(int));

        pthread_mutex_destroy(waiting_for_ack_mutex);
        pthread_cond_destroy(waiting_for_ack_cond);

        munmap(waiting_for_ack_mutex, sizeof(pthread_mutex_t));
        munmap(waiting_for_ack_cond, sizeof(pthread_cond_t));
    }

    Tau_profile_exit_all_threads();
    Tau_destructor_trigger();

    return ret;
}

#if defined(__powerpc64__) || defined(__PPC64__) || defined(__APPLE__)
struct startup_info
{
    void *sda_base;
    int (*main)(int, char **, char **, void *);
    int (*init)(int, char **, char **, void *);
    void (*fini)(void);
};

typedef int (*taupreload_libc_start_main)(int, char **, char **, ElfW(auxv_t) *, void (*)(void), struct startup_info *,
                                          char **);
int __libc_start_main(int argc, char **argv, char **ev, ElfW(auxv_t) * auxvec, void (*rtld_fini)(void),
                      struct startup_info *stinfo, char **stack_on_entry)
#else
typedef int (*taupreload_libc_start_main)(int (*)(int, char **, char **), int, char **, int (*)(int, char **, char **),
                                          void (*)(void), void (*)(void), void *);

int __libc_start_main(int (*_main)(int, char **, char **), int _argc, char **_argv, int (*_init)(int, char **, char **),
                      void (*_fini)(void), void (*_rtld_fini)(void), void *_stack_end)
#endif
{
    // prevent re-entry
    static int _reentry = 0;
    if (_reentry > 0)
        return -1;
    _reentry = 1;

    // get the address of this function
    void *_this_func = __builtin_return_address(0);

    // Save the real main function address
#if defined(__powerpc64__) || defined(__PPC64__) || defined(__APPLE__)
    main_real = stinfo->main;
#else
    main_real = _main;
#endif

    // Find the real __libc_start_main()
    taupreload_libc_start_main user_main = (taupreload_libc_start_main)dlsym(RTLD_NEXT, "__libc_start_main");

    if (user_main && user_main != _this_func)
    {
#if defined(__powerpc64__) || defined(__PPC64__) || defined(__APPLE__)
        struct startup_info my_si;
        my_si.sda_base = stinfo->sda_base;
        my_si.main = taupreload_main;
        my_si.init = stinfo->init;
        my_si.fini = stinfo->fini;

        return user_main(argc, argv, ev, auxvec, rtld_fini, &my_si, stack_on_entry);
#else
        return user_main(taupreload_main, _argc, _argv, _init, _fini, _rtld_fini, _stack_end);
#endif
    }

    else
    {
        fputs("Error! taupreload could not find __libc_start_main!", stderr);
        return -1;
    }
}
