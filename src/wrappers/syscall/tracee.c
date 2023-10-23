#define _GNU_SOURCE

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/time.h>
#include <sys/wait.h>

#include "tracee.h"

#define DEBUG_PTRACE

#ifdef DEBUG_PTRACE
#define DEBUG_PRINT(...)                                                                                               \
    fprintf(stderr, __VA_ARGS__);                                                                                      \
    fflush(stderr);
#else
#define DEBUG_PRINT(...)
#endif

const int init_array_size = 128;
int ending_tracking = 0;

typedef enum
{
    // Stopped by SIGSTOP
    WAIT_STOPPED,
    // Stopped by PTRACE_EVENT_STOP )
    WAIT_STOPPED_NEW_CHILD,
    // Stopped by another signal
    WAIT_STOPPED_OTHER,
    WAIT_SYSCALL,
    WAIT_SYSCALL_EXIT,
    WAIT_SYSCALL_CLONE,
    WAIT_SYSCALL_FORK,
    WAIT_SYSCALL_VFORK,
    WAIT_EXITED,
    WAIT_ERROR
} tracee_wait_t;

static const char *const wait_res_str[WAIT_ERROR + 1] = {
    "WAIT_STOPPED",       "WAIT_STOPPED_NEW_CHILD", "WAIT_STOPPED_OTHER", "WAIT_SYSCALL", "WAIT_SYSCALL_EXIT",
    "WAIT_SYSCALL_CLONE", "WAIT_SYSCALL_FORK",      "WAIT_SYSCALL_VFORK", "WAIT_EXITED",  "WAIT_ERROR"};

typedef struct tracee_thread
{
    pid_t pid;
    int tid;
    // 1 if has entered in a syscall ; 0 otherwise
    int in_syscall;
    // timer for one syscall
    void *syscall_timer;
    // timer for the whole thread life
    void *thread_timer;
} tracee_thread_t;

void print_tracee_thread(tracee_thread_t *tracee)
{
    DEBUG_PRINT("tracee->pid = %d\n", tracee->pid);
    DEBUG_PRINT("tracee->tid = %d\n", tracee->tid);
    DEBUG_PRINT("tracee->in_syscall = %d\n", tracee->in_syscall);
}

tracee_thread_t *create_tracee_thread(pid_t pid, int tid)
{
    tracee_thread_t *new_tracee = (tracee_thread_t *)malloc(sizeof(tracee_thread_t));
    new_tracee->in_syscall = 0;
    new_tracee->pid = pid;
    new_tracee->tid = tid;

    return new_tracee;
}

/**********************************************************
 * Simple dynamic array to store infos on threads to trace *
 ***********************************************************/

typedef struct array_tracee_threads
{
    int capacity;
    int size;
    int lowest_index_free;
    tracee_thread_t **tracee_threads;
} array_tracee_thread_t;

array_tracee_thread_t tracee_threads_array;

void debug_print_array_tracee()
{

    DEBUG_PRINT("tracee_threads_array->capacity = %d\n", tracee_threads_array.capacity);
    DEBUG_PRINT("tracee_threads_array->size = %d\n", tracee_threads_array.size);
    DEBUG_PRINT("tracee_threads_array->lowest_index_free = %d\n", tracee_threads_array.lowest_index_free);
    for (int i = 0; i < tracee_threads_array.size; i++)
    {
        DEBUG_PRINT("[%d] tracee :\n", i);
        print_tracee_thread(tracee_threads_array.tracee_threads[i]);
    }
}

void array_tracee_threads_init()
{
    static volatile int init_done = 0;
    if (!init_done)
    {
        tracee_threads_array.capacity = init_array_size;
        tracee_threads_array.size = 0;
        tracee_threads_array.lowest_index_free = 0;
        tracee_threads_array.tracee_threads = (tracee_thread_t **)malloc(init_array_size * sizeof(tracee_thread_t *));
    }
}

/**
 * @brief Get the tracee_thread for this pid
 *
 * @param pid
 * @return tracee_thread_t
 */
tracee_thread_t *get_tracee_thread(pid_t pid)
{
    for (int i = 0; i < tracee_threads_array.size; i++)
    {
        if (tracee_threads_array.tracee_threads[i])
            if (tracee_threads_array.tracee_threads[i]->pid == pid)
                return tracee_threads_array.tracee_threads[i];
    }
    return NULL;
}

/**
 * @brief Double the capacity of tracee_threads_array
 *
 */
void array_tracee_threads_extend()
{
    tracee_thread_t **tmp = (tracee_thread_t **)realloc(tracee_threads_array.tracee_threads,
                                                        2 * tracee_threads_array.capacity * sizeof(tracee_thread_t *));
}

/**
 * @brief Add a tracee_thread to tracee_threads_array
 *
 * @param pid
 */
void add_tracee_thread(pid_t pid)
{
    int index_free = tracee_threads_array.lowest_index_free;
    DEBUG_PRINT("tracee_threads_array.lowest_index_free = %d\n", tracee_threads_array.lowest_index_free);
    DEBUG_PRINT("tracee_threads_array.size = %d\n", tracee_threads_array.size);

    if (index_free >= tracee_threads_array.capacity)
    {
        array_tracee_threads_extend();
    }
    tracee_threads_array.tracee_threads[index_free] = create_tracee_thread(pid, index_free);
    print_tracee_thread(tracee_threads_array.tracee_threads[index_free]);

    // Update size
    if (index_free == tracee_threads_array.size)
    {
        tracee_threads_array.size++;
    }

    // Update lowest_index_free
    for (int i = index_free; i < tracee_threads_array.size; i++)
    {
        if (tracee_threads_array.tracee_threads[i])
            if (tracee_threads_array.tracee_threads[i]->pid > 0)
            {
                tracee_threads_array.lowest_index_free++;
            }
    }

    DEBUG_PRINT("tracee_threads_array.lowest_index_free = %d\n", tracee_threads_array.lowest_index_free);
    DEBUG_PRINT("tracee_threads_array.size = %d\n", tracee_threads_array.size);
}

/**
 * @brief "Remove" the tracee from the array. In fact just reinitialize it.
 *
 * @param pid
 */
void remove_tracee_thread(pid_t pid)
{
    tracee_thread_t *thread = get_tracee_thread(pid);

    for (int i = 0; i < tracee_threads_array.size; i++)
    {
        if (tracee_threads_array.tracee_threads[i])
        {
            if (tracee_threads_array.tracee_threads[i]->pid == pid)
            {
                tracee_threads_array.tracee_threads[i]->in_syscall = 0;
                tracee_threads_array.tracee_threads[i]->pid = 0;
                if (tracee_threads_array.lowest_index_free > i)
                {
                    tracee_threads_array.lowest_index_free = i;
                }
            }
        }
    }

    // Update size
    for (int i = tracee_threads_array.size - 1; i >= 0; i--)
    {
        if (tracee_threads_array.tracee_threads[i]->pid != 0)
        {
            break;
        }
        tracee_threads_array.size--;
    }
}

/*****************************
 * INTERNAL TRACEE INTERFACE *
 *****************************/

void tracee_handle_stop_syscall(tracee_thread_t *tracee_thread)
{
    if (tracee_thread->in_syscall)
    {
        DEBUG_PRINT("stop call on %d , tid %d\n", tracee_thread->pid, tracee_thread->tid);
        TAU_PROFILER_STOP_TASK(tracee_thread->syscall_timer, tracee_thread->tid);
        tracee_thread->in_syscall = 0;
    }
}

void tracee_stop_all_timers(tracee_thread_t *tracee_thread)
{
    tracee_handle_stop_syscall(tracee_thread);
    TAU_PROFILER_STOP_TASK(tracee_thread->thread_timer, tracee_thread->tid);
}

/**
 * @brief Send PTRACE_INTERRUPT to stop it
 *
 * @param pid
 * @return tracee_error_t
 */
tracee_error_t tracee_interrupt(pid_t pid)
{
    long ret = ptrace(PTRACE_INTERRUPT, pid, NULL, NULL);
    DEBUG_PRINT("PTRACE interrupt on %d\n", pid);

    if (ret < 0)
    {
        perror("ptrace (interrupt)");
        return TRACEE_ERR_OTHER;
    }

    return TRACEE_SUCCESS;
}

/**
 * @brief Send PTRACE_SYSCALL and signal sig to process pid and relaunch it, so the process stops at next entry/exit
 * from a syscall.
 *
 * @param pid
 * @param sig 0 not to send a signal
 * @return tracee_error_t
 */
tracee_error_t tracee_tracksyscalls_ptrace_with_sig(pid_t pid, int sig)
{
    long ret = ptrace(PTRACE_SYSCALL, pid, NULL, sig);
    // DEBUG_PRINT("PTRACE track syscalls on %d with sig %d\n", pid, sig);

    if (ret < 0)
    {
        perror("ptrace (syscall)");
        return TRACEE_ERR_OTHER;
    }

    return TRACEE_SUCCESS;
}

/**
 * @brief Should be used for waiting for a specific pid to STOP, or for waiting for all child (pid = -1)
 * In the last case, waited_tracee is the child tracee_thread which has stopped or exited
 *
 * @param pid -1 to wait for all childs ; > 0 for a specific pid
 * @param waited_tracee is filled if pid = -1 with the child tracee_thread that has stopped/exited
 * @param stop_signal is filled only if the child was stopped for another reason than ptrace handling
 * @return tracee_wait_t
 */
tracee_wait_t tracee_wait_for_child(pid_t pid, tracee_thread_t **waited_tracee, int *stop_signal)
{
    DEBUG_PRINT("waiting on %d\n", pid);
    int child_status;
    pid_t tracee_pid;

    if (pid > 0)
        tracee_pid = waitpid(pid, &child_status, WUNTRACED); // WUNTRACED useful for when attaching the child
    else
        tracee_pid = waitpid(pid, &child_status, 0);

    if (tracee_pid < 0)
    {
        perror("waitpid");
        return WAIT_ERROR;
    }

    // If a specific pid was not specified
    if (waited_tracee)
    {
        *waited_tracee = get_tracee_thread(tracee_pid);
    }

    if (WIFEXITED(child_status))
    {
        DEBUG_PRINT("%d has exited\n", tracee_pid);
        return WAIT_EXITED;
    }

    if (WIFSTOPPED(child_status))
    {
        // The thread may have stopped for several reasons
        // - enter/exit of a SYSCALL
        // - creation of a new thread (clone)
        // - 1st stop of the new thread
        // - SIGTRAP (but no syscall)
        // - SIGSTOP
        // Other reasons

        int wstopsig = WSTOPSIG(child_status);

        // PTRACE_EVENT stops (part 1)
        if (child_status >> 16 == PTRACE_EVENT_STOP)
        {
            // It seems that it's the only case where a PTRACE_EVENT is checked with status>>16 (and not 8)
            DEBUG_PRINT("PTRACE_EVENT_STOP on %d\n", tracee_pid);
            if (wstopsig == SIGTRAP) // When using PTRACE_INTERRUPT
            {
                return WAIT_STOPPED;
            }
            return WAIT_STOPPED_NEW_CHILD;
        }

        // PTRACE_EVENT stops (part 2)
        switch ((child_status >> 8))
        {
        case (SIGTRAP | (PTRACE_EVENT_EXIT << 8)):
            DEBUG_PRINT("PTRACE_EVENT_EXIT on %d\n", tracee_pid);
            return WAIT_SYSCALL_EXIT;

        // Not sure for fork() with TAU : it may create a new node for tau, so we would have to configure it
        // case (SIGTRAP | (PTRACE_EVENT_FORK << 8)):
        // case (SIGTRAP | (PTRACE_EVENT_VFORK << 8)):
        case (SIGTRAP | (PTRACE_EVENT_CLONE << 8)):
            DEBUG_PRINT("PTRACE_EVENT_CLONE on %d\n", tracee_pid);
            // Tracee just called clone()
            pid_t new_tracee_pid;
            // Issue when using tracee_pid?
            ptrace(PTRACE_GETEVENTMSG, (*waited_tracee)->pid, NULL, &new_tracee_pid);
            DEBUG_PRINT("%d created clone %d\n", (*waited_tracee)->pid, new_tracee_pid);

            // The new thread is already tracked and will stop at launch
            add_tracee_thread(new_tracee_pid);
            debug_print_array_tracee();
            return WAIT_SYSCALL_CLONE;
        case (SIGTRAP | (PTRACE_EVENT_STOP << 8)):
            // Seems not to happen. We need to check with status>>16
            DEBUG_PRINT("PTRACE_EVENT_STOP on %d\n", tracee_pid);
            return WAIT_STOPPED_NEW_CHILD;
        default:
            break;
        }

        // No PTRACE_EVENTS
        switch (wstopsig)
        {
        case SIGSTOP:
            DEBUG_PRINT("%d stopped by SIGSTOP\n", tracee_pid);
            return WAIT_STOPPED;

        case (SIGTRAP | 0x80):
            DEBUG_PRINT("%d stopped by SIGTRAP | 0x80\n", tracee_pid);
            return WAIT_SYSCALL;

        case (SIGTRAP):
            DEBUG_PRINT("%d stopped by SIGTRAP\n", tracee_pid);

        default:
            // The tracer will need to relaunch the thread by delivering the same signal which stops it
            if (stop_signal)
            {
                *stop_signal = wstopsig;
            }
            DEBUG_PRINT("Other: %d stopped by sig %d\n", tracee_pid, *stop_signal);
            return WAIT_STOPPED_OTHER;
        }
    }

    // reachable ?
    DEBUG_PRINT("%d Other wait return\n", tracee_pid);
    return WAIT_ERROR;
}

/**
 * @brief This deletes a process from monitoring
 *
 * @param t The tracee to detach from
 * @return tracee_error_t TRACEE_SUCCESS when all ok
 */
tracee_error_t tracee_detach(pid_t pid)
{
    long ret = ptrace(PTRACE_DETACH, pid, NULL, SIGCONT);
    DEBUG_PRINT("PTRACE detach on %d\n", pid);

    if (ret < 0)
    {
        perror("ptrace (detach)");
        return TRACEE_ERR_PERM;
    }
    return TRACEE_SUCCESS;
}

static tracee_error_t tracee_track_syscall(pid_t pid)
{
    /* Syscall detection loop */
    int taskid;
    tracee_error_t ptrace_res;

    ptrace_res = tracee_tracksyscalls_ptrace_with_sig(pid, 0);

    if (ptrace_res != TRACEE_SUCCESS)
    {
        return ptrace_res;
    }

    TAU_PROFILER_CREATE(tracee_threads_array.tracee_threads[0]->thread_timer, "[thread]", "", SYSCALL);
    TAU_PROFILER_START_TASK(tracee_threads_array.tracee_threads[0]->thread_timer, 0);

    while (!ending_tracking)
    {
        int wstopsignal = 0;
        tracee_thread_t *tracee_thread;
        tracee_wait_t wait_res = tracee_wait_for_child(-1, &tracee_thread, &wstopsignal);
        if (!tracee_thread)
        {
            break;
        }
        DEBUG_PRINT("%s on %d\n", wait_res_str[wait_res], tracee_thread->pid);

        switch (wait_res)
        {
        case WAIT_EXITED:
            debug_print_array_tracee();
            tracee_stop_all_timers(tracee_thread);
            remove_tracee_thread(tracee_thread->pid);

            if (tracee_thread->tid == 0)
            {
                return TRACEE_SUCCESS;
            }
            break;

        case WAIT_STOPPED_NEW_CHILD:
            TAU_CREATE_TASK(taskid);

            TAU_PROFILER_CREATE(tracee_thread->thread_timer, "[thread]", "", SYSCALL);
            TAU_PROFILER_START_TASK(tracee_thread->thread_timer, tracee_thread->tid);
            // No break: it will do the WAIT_STOPPED case
        case WAIT_STOPPED_OTHER:
        case WAIT_STOPPED:
            ptrace_res = tracee_tracksyscalls_ptrace_with_sig(tracee_thread->pid, wstopsignal);
            if (ptrace_res != TRACEE_SUCCESS)
            {
                DEBUG_PRINT("Error on %d , tid %d\n", tracee_thread->pid, tracee_thread->tid);
                return ptrace_res;
            }
            break;

        case WAIT_SYSCALL_EXIT:
        case WAIT_SYSCALL_CLONE:
        case WAIT_SYSCALL_FORK:
        case WAIT_SYSCALL_VFORK:
        case WAIT_SYSCALL:
            // Can be the enter or the exit of the syscall
            if (!tracee_thread->in_syscall)
            {
                // Retrieve enter syscall nunber
                int scall_id = get_syscall_id(tracee_thread->pid);

                if (scall_id < 0)
                {
                    DEBUG_PRINT("enter call to %d ??? on %d, tid %d\n", scall_id, tracee_thread->pid,
                                tracee_thread->tid);
                    tracee_tracksyscalls_ptrace_with_sig(tracee_thread->pid, wstopsignal);
                    continue;
                }

                TAU_PROFILER_CREATE(tracee_thread->syscall_timer, get_syscall_name(scall_id), "", SYSCALL);

                DEBUG_PRINT("enter call to %s on %d, tid %d\n", get_syscall_name(scall_id), tracee_thread->pid,
                            tracee_thread->tid);

                tracee_thread->in_syscall = 1;
                TAU_PROFILER_START_TASK(tracee_thread->syscall_timer, tracee_thread->tid);

                /* Continue */
                ptrace_res = tracee_tracksyscalls_ptrace_with_sig(tracee_thread->pid, wstopsignal);
                if (ptrace_res != TRACEE_SUCCESS)
                {
                    tracee_handle_stop_syscall(tracee_thread);

                    DEBUG_PRINT("Error on %d , tid %d\n", tracee_thread->pid, tracee_thread->tid);
                    return ptrace_res;
                }
            }
            else
            {
                tracee_handle_stop_syscall(tracee_thread);

                /* Continue */
                ptrace_res = tracee_tracksyscalls_ptrace_with_sig(tracee_thread->pid, wstopsignal);
                if (ptrace_res != TRACEE_SUCCESS)
                {
                    DEBUG_PRINT("Error on %d , tid %d\n", tracee_thread->pid, tracee_thread->tid);
                    return ptrace_res;
                }
            }
            break;
        default:
            // Error ?
            DEBUG_PRINT("%s on %d\n", wait_res_str[wait_res], tracee_thread->pid);
            break;
        }
    }

    return TRACEE_SUCCESS;
}

static void internal_init_once(void)
{
    static volatile int scall_init_done = 0;

    if (!scall_init_done)
    {
        scalls_init();
        scall_init_done = 1.;
    }
}

/**
 * @brief To use by the parent. Wait for the child to stop itself and seize it.
 *
 * @param pid
 * @return tracee_error_t
 **/
tracee_error_t tracee_seize(pid_t pid)
{ /* Wait for stop */
    tracee_wait_t res = tracee_wait_for_child(pid, NULL, NULL);

    if (res == WAIT_STOPPED)
    {
        long ret = ptrace(PTRACE_SEIZE, pid, NULL, PTRACE_O_TRACESYSGOOD | PTRACE_O_TRACEEXIT | PTRACE_O_TRACECLONE);
        // PTRACE_O_TRACESYSGOOD | PTRACE_O_TRACEEXIT | PTRACE_O_TRACECLONE | PTRACE_O_TRACEFORK | PTRACE_O_TRACEVFORK);
        if (ret < 0)
        {
            perror("ptrace (seize)");
            return TRACEE_ERR_PERM;
        }
        return TRACEE_SUCCESS;
    }
    return TRACEE_ERR_OTHER;
}

void end_tracking(int signum)
{
    DEBUG_PRINT("Signal %d received. Starting end_tracking()\n", signum);
    for (int i = tracee_threads_array.size - 1; i >= 0; i--)
    {
        // For each thread, interrupt the thread and detach it
        tracee_thread_t *tt = tracee_threads_array.tracee_threads[i];
        if (tt->pid > 0)
        {
            // the first thread (tid == 0) is already stopped
            if (i > 0)
            {
                int status;
                waitpid(tt->pid, &status, WNOHANG);

                if (!WIFSTOPPED(status))
                {

                    tracee_interrupt(tt->pid);
                    // wait for the pid to stop
                    tracee_wait_t res = tracee_wait_for_child(tt->pid, NULL, NULL);
                    DEBUG_PRINT("%s on %d\n", wait_res_str[res], tt->pid);
                    
                }
            }
            tracee_stop_all_timers(tt);

            tracee_detach(tt->pid);
            remove_tracee_thread(tt->pid);
        }
    }
    ending_tracking = 1;
    DEBUG_PRINT("end_tracking() done\n");
}

/***************************
 * PUBLIC TRACEE INTERFACE *
 ***************************/

int track_process(pid_t pid)
{
    internal_init_once();
    array_tracee_threads_init();
    add_tracee_thread(pid);

    struct sigaction sa;

    sa.sa_handler = end_tracking;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGRTMIN, &sa, NULL) == -1)
    {
        perror("sigaction");
    }

    DEBUG_PRINT("Will track %d (thread %d)\n", pid, 0);

    // The child is supposed to use prepare_to_be_tracked()
    tracee_error_t res = tracee_seize(pid);

    if (res != TRACEE_SUCCESS)
    {
        // It can be because of permission or because the process is alredy tracked by an other tracer
        perror(tracee_error_str[res]);
        return 1;
    }

    DEBUG_PRINT("%d (thread %d) attached\n", pid, 0);

    // The child_thread is attached and currently stopped
    tracee_error_t ptrace_res = tracee_track_syscall(pid);
    if (ptrace_res != TRACEE_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    DEBUG_PRINT("tracking_done\n");

    // Wait the child to exit
    int status;
    waitpid(pid, &status, 0);

    DEBUG_PRINT("Child just exited\n");
    return EXIT_SUCCESS;
}

void prepare_to_be_tracked(pid_t pid)
{
    // ptrace(PTRACE_TRACEME, 0, NULL, NULL);
    // Permission issues even with the parent so we use prctl() to set the permission

    if (prctl(PR_SET_PTRACER, pid, NULL, NULL, NULL, NULL) < 0)
    {
        perror("prctl");
    }

    DEBUG_PRINT("%d just set ptracer as %d\n", getpid(), pid);
    raise(SIGSTOP);
    DEBUG_PRINT("%d is attached\n", getpid());
}
