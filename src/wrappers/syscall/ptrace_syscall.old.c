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
    task_creator_thread_tid = (volatile int *)mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);

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

    pthread_mutexattr_settype(&mutex_attr, PTHREAD_MUTEX_ERRORCHECK); // PTHREAD_MUTEX_RECURSIVE

    pthread_mutex_init(waiting_for_ack_mutex, &mutex_attr);
    pthread_cond_init(waiting_for_ack_cond, &cond_attr);

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
        munmap((int *) task_creator_thread_tid, sizeof(int));

        pthread_mutex_destroy(waiting_for_ack_mutex);
        pthread_cond_destroy(waiting_for_ack_cond);
        munmap(waiting_for_ack_mutex, sizeof(pthread_mutex_t));
        munmap(waiting_for_ack_cond, sizeof(pthread_cond_t));

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

    pthread_join(task_creator_thread, NULL);

    TAU_PROFILER_STOP(handle);

    Tau_profile_exit_all_threads();
    Tau_destructor_trigger();
}
