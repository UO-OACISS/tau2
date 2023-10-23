void __attribute__((constructor)) taupreload_init(void);
void __attribute__((destructor)) taupreload_fini(void);

#include <TAU.h>
#include <stdlib.h>

#include <tracee.h>

#if defined(__GNUC__)
#define __TAU_FUNCTION__ __PRETTY_FUNCTION__
#else
#define __TAU_FUNCTION__ __func__
#endif

extern void Tau_init_initializeTAU(void);
extern void Tau_profile_exit_all_threads(void);

void *handle;

void taupreload_init()
{
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
        
        // Tell parent to stop the tracking
        kill(getppid(), SIGRTMIN); // SIGUSR already taken by TAU
    }
    else
    {
        TAU_PROFILE_SET_CONTEXT(1);
        /* Parent */
        track_process(rpid);

        Tau_profile_exit_all_threads();
        Tau_destructor_trigger();
        exit(0);
    }
}

void taupreload_fini()
{
    TAU_PROFILER_STOP(handle);
    Tau_profile_exit_all_threads();
    Tau_destructor_trigger();
}
