void __attribute__((constructor)) taupreload_init(void);
void __attribute__((destructor)) taupreload_fini(void);

#include <TAU.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

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

extern void Tau_init_initializeTAU(void);
extern void Tau_profile_exit_all_threads(void);

void *handle;

void taupreload_init()
{
    shared_num_tasks = (volatile int *)mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    waiting_for_ack = (volatile int *)mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    parent_has_dumped = (volatile int *)mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    task_creater_thread_tid = (volatile int *)mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

    *shared_num_tasks = 0;
    *waiting_for_ack = 0;
    *parent_has_dumped = 0;
    *task_creater_thread_tid = -1;

    pid_t rpid = fork();

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
        TAU_PROFILER_CREATE(handle, __TAU_FUNCTION__, "", TAU_DEFAULT);
        TAU_PROFILER_START(handle);

        prepare_to_be_tracked(getppid());
    }
    else
    {
        /* Parent */
        track_process(rpid);

        munmap((int *) shared_num_tasks, sizeof(int));
        munmap((int *) waiting_for_ack, sizeof(int));
        munmap((int *) parent_has_dumped, sizeof(int));
        munmap((int *) task_creater_thread_tid, sizeof(int));

        Tau_profile_exit_all_threads();
        Tau_destructor_trigger();

        exit(0);
    }
}

void taupreload_fini()
{
    // Tell parent to stop the tracking
    kill(getppid(), SIG_STOP_PTRACE);
    DEBUG_PRINT("%d just sent signal SIG_STOP_PTRACE to %d\n", getpid(), getppid());

    while (!*parent_has_dumped)
    {
    }

    pthread_join(task_creater_thread, NULL);

    TAU_PROFILER_STOP(handle);

    Tau_profile_exit_all_threads();
    Tau_destructor_trigger();
}
