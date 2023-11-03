#define _GNU_SOURCE

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/time.h>
#include <sys/wait.h>

#include "tracee.h"

// gettid() and tgkill()
#if !(__GLIBC_PREREQ(2, 30))
#include <sys/syscall.h>

pid_t gettid(void)
{
    return syscall(SYS_gettid);
}

// cf man tkill
int tgkill(pid_t tgid, pid_t tid, int sig)
{
    return syscall(SYS_tkill, tid, sig);
}
#endif

// To disable ptrace on the threads of the child
// #define NO_TRACECLONE

// #define DEBUG_PTRACE

#ifdef DEBUG_PTRACE
#define DEBUG_PRINT(...)                                                                                               \
    fprintf(stderr, __VA_ARGS__);                                                                                      \
    fflush(stderr);
#else
#define DEBUG_PRINT(...)
#endif

// expands to "return TRACEE_ERR_PARAM" if the parameter is null
#define CHECK_IF_PARAM_IS_NULL(tt)                                                                                     \
    if (!tt)                                                                                                           \
    {                                                                                                                  \
        fprintf(stderr, "ERROR : %s, parameter is NULL\n", __func__);                                                  \
        fflush(stderr);                                                                                                \
        return TRACEE_ERR_PARAM;                                                                                       \
    }

#define CHECK_ERROR_ON_PTRACE_RES(ptrace_res, tracee_thread)                                                           \
    if (ptrace_res != TRACEE_SUCCESS)                                                                                  \
    {                                                                                                                  \
        DEBUG_PRINT("Error on %d , tid %d\n", tracee_thread->pid, tracee_thread->tid);                                 \
        return ptrace_res;                                                                                             \
    }

const int init_array_size = 128;
int ending_tracking = 0;

int local_num_tasks = 0;
pthread_t task_creator_thread;

// Are initialized with mmap in ptrace_syscall.c

volatile int *task_creator_thread_tid;
volatile int *shared_num_tasks;
volatile int *waiting_for_ack;
volatile int *parent_has_dumped;

pthread_mutex_t *waiting_for_ack_mutex;
pthread_cond_t *waiting_for_ack_cond;

// To use before setting any timer on a (fake) thread
// Otherwise, it will use the gpu_timers for the fake threads
extern void Tau_set_fake_thread_use_cpu_metric(int tid);

typedef enum
{
    // Stopped by SIGSTOP
    WAIT_STOPPED,
    // Stopped by PTRACE_EVENT_STOP
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
    // 1 if this thread is currently stopped
    int is_stopped;
    // 1 if has entered in a syscall ; 0 otherwise
    int in_syscall;
    // 1 if it is a new child to start
    int is_new_child;
    // 1 if it has send a signal to increment num_task
    int has_sent_signal;
    // timer for one syscall
    void *syscall_timer;
} tracee_thread_t;

static void print_tracee_thread(tracee_thread_t *tracee)
{
    if (!tracee)
    {
        return;
    }
    DEBUG_PRINT("tracee->pid = %d\n", tracee->pid);
    DEBUG_PRINT("tracee->tid = %d\n", tracee->tid);
    DEBUG_PRINT("tracee->in_syscall = %d\n", tracee->in_syscall);
    DEBUG_PRINT("tracee->is_stopped = %d\n", tracee->is_stopped);
    DEBUG_PRINT("tracee->is_new_child = %d\n", tracee->is_new_child);
    DEBUG_PRINT("tracee->has_sent_signal = %d\n", tracee->has_sent_signal);
}

static tracee_thread_t *create_tracee_thread(pid_t pid, int tid)
{
    tracee_thread_t *new_tracee = (tracee_thread_t *)malloc(sizeof(tracee_thread_t));
    new_tracee->is_stopped = 1;
    new_tracee->in_syscall = 0;
    new_tracee->is_new_child = 1;
    new_tracee->has_sent_signal = 0;
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

static void debug_print_array_tracee(void)
{
    DEBUG_PRINT("tracee_threads_array->capacity = %d\n", tracee_threads_array.capacity);
    DEBUG_PRINT("tracee_threads_array->size = %d\n", tracee_threads_array.size);
    DEBUG_PRINT("tracee_threads_array->lowest_index_free = %d\n", tracee_threads_array.lowest_index_free);
    for (int i = 0; i < tracee_threads_array.size; i++)
    {
        DEBUG_PRINT("[%d] tracee :\n", i);
        if (tracee_threads_array.tracee_threads[i])
            print_tracee_thread(tracee_threads_array.tracee_threads[i]);
    }
}

static void array_tracee_threads_init(void)
{
    static volatile int array_init_done = 0;
    if (array_init_done)
    {
        return;
    }

    tracee_threads_array.capacity = init_array_size;
    tracee_threads_array.size = 0;
    tracee_threads_array.lowest_index_free = 0;
    tracee_threads_array.tracee_threads = (tracee_thread_t **)malloc(init_array_size * sizeof(tracee_thread_t *));
}

static void array_tracee_threads_free(void)
{
    for (int i = tracee_threads_array.size - 1; i >= 0; i--)
    {
        if (tracee_threads_array.tracee_threads[i])
        {
            free(tracee_threads_array.tracee_threads[i]);
            tracee_threads_array.tracee_threads[i] = NULL;
        }
    }
    free(tracee_threads_array.tracee_threads);
    tracee_threads_array.tracee_threads = NULL;
}

/**
 * @brief Get the tracee_thread for this pid
 *
 * @param pid
 * @return tracee_thread_t
 */
static tracee_thread_t *get_tracee_thread(pid_t pid)
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
static void array_tracee_threads_extend(void)
{
    tracee_thread_t **tmp = (tracee_thread_t **)realloc(tracee_threads_array.tracee_threads,
                                                        2 * tracee_threads_array.capacity * sizeof(tracee_thread_t *));
    tracee_threads_array.tracee_threads = tmp;
    tracee_threads_array.capacity *= 2;
}

/**
 * @brief Add a tracee_thread to tracee_threads_array.
 *
 * @param pid
 * @param tid
 *
 * @note tid is set in tracee_start_tracking_tt(), so we may remove the tid parameter from add_tracee_thread() and put
 * it to -1 by default
 */
static tracee_thread_t *add_tracee_thread(pid_t pid, int tid)
{
    if (tracee_threads_array.lowest_index_free > tracee_threads_array.size)
        tracee_threads_array.lowest_index_free = tracee_threads_array.size;

    int index_free = tracee_threads_array.lowest_index_free;
    DEBUG_PRINT("tracee_threads_array.lowest_index_free = %d\n", tracee_threads_array.lowest_index_free);
    DEBUG_PRINT("tracee_threads_array.size = %d\n", tracee_threads_array.size);

    if (index_free >= tracee_threads_array.capacity)
    {
        array_tracee_threads_extend();
    }
    tracee_thread_t *tt = create_tracee_thread(pid, tid);
    tracee_threads_array.tracee_threads[index_free] = tt;
    print_tracee_thread(tt);

    // Update size
    if (index_free == tracee_threads_array.size)
    {
        tracee_threads_array.size++;
    }

    // Update lowest_index_free
    for (int i = index_free; i < tracee_threads_array.size; i++)
    {
        if (tracee_threads_array.tracee_threads[i])
            tracee_threads_array.lowest_index_free++;
    }

    DEBUG_PRINT("tracee_threads_array.lowest_index_free = %d\n", tracee_threads_array.lowest_index_free);
    DEBUG_PRINT("tracee_threads_array.size = %d\n", tracee_threads_array.size);
    return tt;
}

/**
 * @brief Remove the tracee from the array
 *
 * @param pid
 */
static void remove_tracee_thread(pid_t pid)
{
    tracee_thread_t *thread = get_tracee_thread(pid);

    for (int i = 0; i < tracee_threads_array.size; i++)
    {
        if (tracee_threads_array.tracee_threads[i])
        {
            if (tracee_threads_array.tracee_threads[i]->pid == pid)
            {
                free(tracee_threads_array.tracee_threads[i]);
                tracee_threads_array.tracee_threads[i] = NULL;
            }
        }
    }
    // Update size
    for (int i = tracee_threads_array.size - 1; i >= 0; i--)
    {
        if (tracee_threads_array.tracee_threads[i])
        {
            break;
        }
        tracee_threads_array.size--;
    }
    if (tracee_threads_array.lowest_index_free > tracee_threads_array.size)
        tracee_threads_array.lowest_index_free = tracee_threads_array.size;
}

/*****************************
 * INTERNAL TRACEE INTERFACE *
 *****************************/

/**
 * @brief Should be used for waiting for a specific pid to STOP, or for waiting for any child (pid = -1)
 * Return in waited_tracee the child tracee_thread which has stopped or exited.
 *
 * @param pid -1 to wait for all childs ; > 0 for a specific pid
 * @param waited_tracee is filled with the tracee_thread_t which has stopped or exited
 * @param stop_signal is filled only if the child was stopped for another reason than ptrace handling, in order to
 * inject the signal back when restarting the thread
 * @return tracee_wait_t
 */
static tracee_wait_t tracee_wait_for_child(pid_t pid, tracee_thread_t **waited_tracee, int *stop_signal)
{
    DEBUG_PRINT("waiting on %d\n", pid);
    int child_status;
    pid_t tracee_pid = 0;

    if (pid > 0)
        tracee_pid = waitpid(pid, &child_status, WUNTRACED); // WUNTRACED useful for when attaching the child
    else
    {
        tracee_pid = waitpid(pid, &child_status, 0);
    }

    if (tracee_pid < 0)
    {
        perror("waitpid");
        return WAIT_ERROR;
    }

    if (waited_tracee)
    {
        if (!(*waited_tracee))
        {
            *waited_tracee = get_tracee_thread(tracee_pid);
            if (!(*waited_tracee))
            {
                // It happens when the CLONE_EVENT notification happens after the STOPPED_NEW_CHILD notification
                *waited_tracee = add_tracee_thread(tracee_pid, -1);
                return WAIT_STOPPED_NEW_CHILD;
            }
        }
        (*waited_tracee)->is_stopped = 1;
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
            // Should not be reached anymore because we already managed this case some lines above
            return WAIT_STOPPED_NEW_CHILD;
        }

        // PTRACE_EVENT stops (part 2)
        // None of these are currently used and are treated as any WAIT_SYSCALL in the main loop
        // Still useful for debugging
        switch ((child_status >> 8))
        {
        case (SIGTRAP | (PTRACE_EVENT_EXIT << 8)):
            DEBUG_PRINT("PTRACE_EVENT_EXIT on %d\n", tracee_pid);
            return WAIT_SYSCALL_EXIT;
        // Not sure for fork() with TAU : it may create a new node for tau, so we would have to configure it
        case (SIGTRAP | (PTRACE_EVENT_FORK << 8)):
            DEBUG_PRINT("PTRACE_EVENT_FORK on %d\n", tracee_pid);
            return WAIT_SYSCALL_FORK;
        case (SIGTRAP | (PTRACE_EVENT_VFORK << 8)):
            DEBUG_PRINT("PTRACE_EVENT_VFORK on %d\n", tracee_pid);
            return WAIT_SYSCALL_VFORK;
        case (SIGTRAP | (PTRACE_EVENT_CLONE << 8)):
            /* Since the new cloned is automatically tracked, we manage it just after the wait */
            DEBUG_PRINT("PTRACE_EVENT_CLONE on %d\n", tracee_pid);
            return WAIT_SYSCALL_CLONE;
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

        default:
            // The tracer will need to relaunch the thread by delivering the same signal which stops it
            if (stop_signal)
            {
                *stop_signal = wstopsig;
            }
            DEBUG_PRINT("Other: %d stopped by sig %d\n", tracee_pid, wstopsig);
            return WAIT_STOPPED_OTHER;
        }
    }

    // reachable ?
    DEBUG_PRINT("%d Other wait return\n", tracee_pid);
    return WAIT_ERROR;
}

static tracee_error_t tracee_handle_stop_syscall(tracee_thread_t *tracee_thread)
{
    CHECK_IF_PARAM_IS_NULL(tracee_thread);
    CHECK_IF_PARAM_IS_NULL(tracee_thread->syscall_timer);
    if (tracee_thread->in_syscall)
    {
        DEBUG_PRINT("stop call on %d , tid %d\n", tracee_thread->pid, tracee_thread->tid);
        TAU_PROFILER_STOP_TASK(tracee_thread->syscall_timer, tracee_thread->tid);
        tracee_thread->in_syscall = 0;
    }
    return TRACEE_SUCCESS;
}

/**
 * @brief Send PTRACE_INTERRUPT to stop it
 *
 * @param pid
 * @return tracee_error_t
 */
static tracee_error_t tracee_interrupt(tracee_thread_t *tt)
{
    CHECK_IF_PARAM_IS_NULL(tt);
    long ret = ptrace(PTRACE_INTERRUPT, tt->pid, NULL, NULL);
    DEBUG_PRINT("PTRACE interrupt on %d\n", tt->pid);

    if (ret < 0)
    {
        perror("ptrace (interrupt)");
        return TRACEE_ERR_OTHER;
    }

    return TRACEE_SUCCESS;
}

/**
 * @brief Send PTRACE_CONT to continue without tracking the syscall. (not used)
 *
 * @param pid
 * @return tracee_error_t
 */
static tracee_error_t tracee_continue(tracee_thread_t *tt, int sig)
{
    CHECK_IF_PARAM_IS_NULL(tt);
    long ret = ptrace(PTRACE_CONT, tt->pid, NULL, sig);
    DEBUG_PRINT("PTRACE continue on %d\n", tt->pid);

    if (ret < 0)
    {
        perror("ptrace (continue)");
        return TRACEE_ERR_OTHER;
    }
    tt->is_stopped = 0;

    return TRACEE_SUCCESS;
}

/**
 * @brief This deletes a process from monitoring
 *
 * @param t The tracee to detach from
 * @return tracee_error_t TRACEE_SUCCESS when all ok
 */
static tracee_error_t tracee_detach(tracee_thread_t *tt)
{
    CHECK_IF_PARAM_IS_NULL(tt);
    // (I don't understand why it does not work without SIGCONT)
    long ret = ptrace(PTRACE_DETACH, tt->pid, NULL, SIGCONT);
    DEBUG_PRINT("PTRACE detach on %d, tid %d\n", tt->pid, tt->tid);

    if (ret < 0)
    {
        perror("ptrace (detach)");
        return TRACEE_ERR_PERM;
    }

    return TRACEE_SUCCESS;
}

/**
 * @brief To use by the parent. Wait for the child to stop itself and seize it.
 *
 * @param pid
 * @return tracee_error_t
 **/
static tracee_error_t tracee_seize(tracee_thread_t *tt)
{ /* Wait for stop */
    CHECK_IF_PARAM_IS_NULL(tt);
    tracee_wait_t res = tracee_wait_for_child(tt->pid, &tt, NULL);

    if (res == WAIT_STOPPED)
    {
        /* Possible options:
         *  - PTRACE_O_TRACESYSGOOD to track syscalls
         *  - PTRACE_O_TRACEEXIT to track the exit
         *  - PTRACE_O_TRACECLONE to track the childs threads of the main process to track
         *  - PTRACE_O_TRACEFORK / PTRACE_O_TRACEVFORK to track the childs threads made with fork/vfork
         */

#ifdef NO_TRACECLONE
        long ret = ptrace(PTRACE_SEIZE, tt->pid, NULL, PTRACE_O_TRACESYSGOOD | PTRACE_O_TRACEEXIT);
#else
        long ret =
            ptrace(PTRACE_SEIZE, tt->pid, NULL, PTRACE_O_TRACESYSGOOD | PTRACE_O_TRACEEXIT | PTRACE_O_TRACECLONE);
#endif
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

/**
 * @brief Send PTRACE_SYSCALL and signal sig to process pid and relaunch it, so the process stops at next entry/exit
 * from a syscall.
 *
 * @param pid
 * @param sig 0 not to send a signal
 * @return tracee_error_t
 */
static tracee_error_t tracee_tracksyscalls_ptrace_with_sig(tracee_thread_t *tt, int sig)
{
    CHECK_IF_PARAM_IS_NULL(tt);
    long ret = ptrace(PTRACE_SYSCALL, tt->pid, NULL, sig);
    // DEBUG_PRINT("PTRACE track syscalls on %d with sig %d\n", pid, sig);

    if (ret < 0)
    {
        perror("ptrace (syscall)");
        return TRACEE_ERR_OTHER;
    }
    tt->is_stopped = 0;

    return TRACEE_SUCCESS;
}

/**
 * @brief Put the correct tid on the thread.
 *        And restart the thread by calling tracee_tracksyscalls_ptrace_with_sig on it and by starting the thread timer
 * 
 * @note Be sure that no one else is using shared_num_tasks when entering this function
 *
 * @param tt
 * @return tracee_error_t
 */
static tracee_error_t tracee_start_tracking_tt(tracee_thread_t *tt)
{
    
    CHECK_IF_PARAM_IS_NULL(tt);

    DEBUG_PRINT("Will update local_num_tasks. local_num_tasks = %d, shared_num_tasks = %d\n", local_num_tasks,
                *shared_num_tasks);
    while (local_num_tasks < *shared_num_tasks)
    {
        // Create false/empty tid to reserve them for the other TAU runtime
        TAU_CREATE_TASK(local_num_tasks);
    }
    DEBUG_PRINT("Has updated local_num_tasks. local_num_tasks = %d, shared_num_tasks = %d\n", local_num_tasks,
                *shared_num_tasks);

    // As explained in the task creator routine, local_num_tasks and *shared_num_tasks are now the reserved task id for
    // the syscalls of the new child
    tt->tid = local_num_tasks;
    Tau_set_fake_thread_use_cpu_metric(tt->tid);
    Tau_create_top_level_timer_if_necessary_task(tt->tid);
    tracee_error_t ptrace_res = tracee_tracksyscalls_ptrace_with_sig(tt, 0);
    CHECK_ERROR_ON_PTRACE_RES(ptrace_res, tt);
    tt->is_new_child = 0;
    DEBUG_PRINT("%d (thread %d) attached and tracked\n", tt->pid, tt->tid);
    return ptrace_res;
}

/**
 * @brief Return the oldest tracee which has not been start and has sent a signal.
 * If there are none, return the oldest tracee which has not sent a signal.
 * Else, return NULL
 *
 * @return tracee_thread_t*
 */
static tracee_thread_t *get_waiting_new_child_tt(void)
{
    int i_has_sent = -1;
    int i_new_child = -1;
    for (int i = 0; i < tracee_threads_array.size; i++)
    {
        tracee_thread_t *tt = tracee_threads_array.tracee_threads[i];
        if (tt)
        {
            if (tt->is_new_child)
            {
                if (tt->has_sent_signal)
                {
                    i_has_sent = i;
                    break;
                }
                if (i_new_child < 0)
                {
                    i_new_child = i;
                }
            }
        }
    }
    if (i_has_sent > 0)
    {
        return tracee_threads_array.tracee_threads[i_has_sent];
    }
    if (i_new_child > 0)
    {
        return tracee_threads_array.tracee_threads[i_new_child];
    }
    return NULL;
}

/* Syscall detection loop */
static tracee_error_t tracee_track_syscall(tracee_thread_t *tt)
{
    CHECK_IF_PARAM_IS_NULL(tt);
    tracee_error_t ptrace_res;

    // Just pretend that the main child has made the same steps as would do any other child to start it
    tt->has_sent_signal = 1;
    ptrace_res = tracee_start_tracking_tt(tt);
    CHECK_ERROR_ON_PTRACE_RES(ptrace_res, tt);

    while (!ending_tracking)
    {
        int wstopsignal = 0;
        tracee_thread_t *tracee_thread = NULL;
        tracee_wait_t wait_res = tracee_wait_for_child(-1, &tracee_thread, &wstopsignal);

        if (ending_tracking)
        {
            break;
        }

        // The idea is to wait for the main child to update its local threads counter after having created a child
        // Until then, the new thread created by clone() will be waiting
        int mutex_res = pthread_mutex_trylock(waiting_for_ack_mutex);
        if (mutex_res == 0)
        {
            if (*waiting_for_ack == 0)
            {
                tracee_thread_t *waiting_new_child_tt = get_waiting_new_child_tt();
                if (waiting_new_child_tt)
                {
                    if (!(waiting_new_child_tt->has_sent_signal))
                    {
                        // We need to wake up the task creator to increment and synchronize num_task
                        DEBUG_PRINT("Will wake up the task creator for child %d\n", waiting_new_child_tt->pid);
                        waiting_new_child_tt->has_sent_signal = 1;
                        *waiting_for_ack = 1;
                        pthread_cond_signal(waiting_for_ack_cond);
                    }
                    else
                    {
                        DEBUG_PRINT("Wait for the task creator finished: will start %d\n", waiting_new_child_tt->pid);
                        ptrace_res = tracee_start_tracking_tt(waiting_new_child_tt);
                        CHECK_ERROR_ON_PTRACE_RES(ptrace_res, waiting_new_child_tt);
                    }
                }
            }
            pthread_mutex_unlock(waiting_for_ack_mutex);
        }
        else
        {
            // The task creator may be running
            if (mutex_res != EBUSY)
            {
                DEBUG_PRINT("Error with mutex_lock that returns %d\n", mutex_res);
                perror("mutex");
                ending_tracking = 1;
                break;
            }
            DEBUG_PRINT("Trylock mutex failed because the task creator is using it\n");
        }

        if (!tracee_thread)
        {
            break;
        }

        DEBUG_PRINT("%s on %d\n", wait_res_str[wait_res], tracee_thread->pid);

        switch (wait_res)
        {
        case WAIT_EXITED:
            debug_print_array_tracee();
            tracee_handle_stop_syscall(tracee_thread);

            int is_main_child = (tt == tracee_thread ? 1 : 0);

            remove_tracee_thread(tracee_thread->pid);

            if (is_main_child)
            {
                ending_tracking = 1;
            }
            break;

        case WAIT_STOPPED_NEW_CHILD:
            // The child is put in a waiting state and will be restart when we are sure for the tid to set
            break;
        case WAIT_STOPPED_OTHER:
        case WAIT_STOPPED:
            ptrace_res = tracee_tracksyscalls_ptrace_with_sig(tracee_thread, wstopsignal);
            CHECK_ERROR_ON_PTRACE_RES(ptrace_res, tracee_thread);
            break;

        // All of these are curently treated as any syscall
        case WAIT_SYSCALL_CLONE:
        case WAIT_SYSCALL_FORK:
        case WAIT_SYSCALL_VFORK:
        case WAIT_SYSCALL_EXIT:
        case WAIT_SYSCALL:
            // Can be the enter or the exit of the syscall
            if (!tracee_thread->in_syscall)
            {
                // Retrieve enter syscall nunber
                int scall_id = get_syscall_id(tracee_thread->pid);

                if (scall_id < 0)
                {
                    // Not necessary an error
                    DEBUG_PRINT("enter call to %d (?) on %d, tid %d\n", scall_id, tracee_thread->pid,
                                tracee_thread->tid);
                    ptrace_res = tracee_tracksyscalls_ptrace_with_sig(tracee_thread, wstopsignal);
                    CHECK_ERROR_ON_PTRACE_RES(ptrace_res, tracee_thread);
                    continue;
                }

                TAU_PROFILER_CREATE(tracee_thread->syscall_timer, get_syscall_name(scall_id), "", SYSCALL);

                DEBUG_PRINT("enter call to %s on %d, tid %d\n", get_syscall_name(scall_id), tracee_thread->pid,
                            tracee_thread->tid);

                tracee_thread->in_syscall = 1;
                TAU_PROFILER_START_TASK(tracee_thread->syscall_timer, tracee_thread->tid);
            }
            else
            {
                // Exit of syscall
                tracee_handle_stop_syscall(tracee_thread);
            }

            /* Continue */
            ptrace_res = tracee_tracksyscalls_ptrace_with_sig(tracee_thread, wstopsignal);
            CHECK_ERROR_ON_PTRACE_RES(ptrace_res, tracee_thread);
            break;
        default:
            // Error ?
            DEBUG_PRINT("%s on %d\n", wait_res_str[wait_res], tracee_thread->pid);
            ending_tracking = 1;
            break;
        } // end switch
    }

    return TRACEE_SUCCESS;
}

static tracee_error_t tracee_detach_everything(void)
{
    // Detachment of all tracked threads
    DEBUG_PRINT("Will detach every traced threads\n");
    debug_print_array_tracee();
    for (int i = tracee_threads_array.size - 1; i >= 0; i--)
    {
        // For each thread, interrupt the thread and detach it
        tracee_thread_t *tt = tracee_threads_array.tracee_threads[i];
        if (tt)
        {
            if (!(tt->is_stopped))
            {
                tracee_interrupt(tt);
                // wait for the pid to stop
                tracee_wait_t res = tracee_wait_for_child(tt->pid, &tt, NULL);
                DEBUG_PRINT("%s on %d\n", wait_res_str[res], tt->pid);
            }

            tracee_detach(tt);
            remove_tracee_thread(tt->pid);
        }
    }
    return TRACEE_SUCCESS;
}

/**
 * @brief Signal the parent to stop the tracking
 *
 * @param signum signal received (cf sigaction)
 */
static void end_tracking(int signum)
{
    // The signal should be sent by the main child
    DEBUG_PRINT("Signal %d received. Starting end_tracking()\n", signum);
    ending_tracking = 1;
}

/**
 * @brief Routine for a thread whose only task is to create false tid in TAU when needed
 */
static void *tau_task_creator(void *ptr)
{
    *task_creator_thread_tid = gettid();
    DEBUG_PRINT("Task creator thread created with tid = %d\n", gettid());

    while (!(*parent_has_dumped))
    {
        DEBUG_PRINT("Routine: trying to lock mutex\n");
        // Possible issue:
        // tpp.c:82: __pthread_tpp_change_priority: Assertion `new_prio == -1 || (new_prio >= fifo_min_prio
        // && new_prio <= fifo_max_prio)' failed.
        // Update: it seems the error is due forgetting to initialize properly mutex_attr in ptrace_syscall.c
        int mutex_res = pthread_mutex_lock(waiting_for_ack_mutex);
        if (mutex_res < 0)
        {
            perror("Routine: mutex");
            kill(getppid(), SIG_STOP_PTRACE);
            break;
        }
        DEBUG_PRINT("Routine: will wait\n");
        while ((!(*waiting_for_ack)) && (!(*parent_has_dumped)))
        {
            mutex_res = pthread_cond_wait(waiting_for_ack_cond, waiting_for_ack_mutex);
            if (mutex_res < 0)
            {
                perror("Routine: cond_wait");
                kill(getppid(), SIG_STOP_PTRACE);
                return NULL;
            }
        }
        DEBUG_PRINT("Routine: wait done\n");
        if (*parent_has_dumped)
        {
            break;
        }

        DEBUG_PRINT("Routine: Task creator thread will create a task\n");
        DEBUG_PRINT("Routine: Before TAU_CREATE_TASK(local_num_tasks). local_num_tasks = %d, shared_num_tasks = %d\n",
                    local_num_tasks, *shared_num_tasks);
        // Create false/empty tid to reserve them for the other TAU runtime
        TAU_CREATE_TASK(local_num_tasks);
        // local_num_tasks is now the task id to use for the syscall of the new child and also the total number of
        // thread for this TAU runtime

        // The main child may have created some tids before updating local_num_threads
        *shared_num_tasks = local_num_tasks;
        *waiting_for_ack = 0;

        DEBUG_PRINT("Routine: ending update_thread_nb(). local_num_tasks = %d, shared_num_tasks = %d\n",
                    local_num_tasks, *shared_num_tasks);
        pthread_mutex_unlock(waiting_for_ack_mutex);
    }
    DEBUG_PRINT("Ending task creator routine\n");
    return NULL;
}

static void internal_init_once(void)
{
    struct sigaction sa;

    sa.sa_handler = end_tracking;
    sa.sa_flags = SA_RESTART;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIG_STOP_PTRACE, &sa, NULL) == -1)
    {
        perror("sigaction");
    }

    static volatile int init_done = 0;

    if (init_done)
    {
        return;
    }

    scalls_init();
    array_tracee_threads_init();

    init_done = 1.;
}

/***************************
 * PUBLIC TRACEE INTERFACE *
 ***************************/

int track_process(pid_t pid)
{
    internal_init_once();

    // Here we use a dummy tid number
    tracee_thread_t *tt = add_tracee_thread(pid, -1);

    // The child is supposed to use prepare_to_be_tracked()
    tracee_error_t res = tracee_seize(tt);
    if (res != TRACEE_SUCCESS)
    {
        // It can be because of permission or because the process is alredy tracked by another tracer
        perror(tracee_error_str[res]);
        Tau_destructor_trigger();
        *parent_has_dumped = 1;
        pthread_cond_signal(waiting_for_ack_cond);
        return EXIT_FAILURE;
    }

    // The child_thread is attached and currently stopped
    res = tracee_track_syscall(tt);

    // We cannot not dump the profile.0.0.0 file, so we force the parent to dump everything before the child
    // This way, the child will replace the files like profile.0.0.0 by its ones
    Tau_destructor_trigger();
    *parent_has_dumped = 1;
    pthread_cond_signal(waiting_for_ack_cond);

    res = tracee_detach_everything();

    DEBUG_PRINT("tracking_done\n");
    array_tracee_threads_free();

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

    DEBUG_PRINT("%d [%d] just set ptracer as %d\n", getpid(), gettid(), pid);

    pthread_create(&task_creator_thread, NULL, tau_task_creator, (void *)NULL);

    // We create a false tid, which will not be used, for the parent
    TAU_CREATE_TASK(local_num_tasks);
    // And update shared_num_tasks
    *shared_num_tasks = local_num_tasks;

    // Make sure the thread is launched before raise(SIGSTOP)
    while (*task_creator_thread_tid < 0)
    {
        // Lazy solution to a minor problem
        usleep(100);
    }

    // tgkill(getpid(), getpid(), SIGSTOP); // Doesn't work as we want to, hence the following lines

    raise(SIGSTOP);
    // The previous SIGSTOP stopped the pthread, so we need to relaunch it
    tgkill(getpid(), *task_creator_thread_tid, SIGCONT);
    DEBUG_PRINT("%d is attached\n", getpid());
}
