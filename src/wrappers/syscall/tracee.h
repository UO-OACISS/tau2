#ifndef TRACEE_H
#define TRACEE_H

#include <errno.h>
#include <signal.h>
#include <sys/ptrace.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <Profile/Profiler.h>

#include "scalls.h"

#define SYSCALL TAU_USER

// Signal for the child to send to the parent to stop the tracking with ptrace
#define SIG_STOP_PTRACE SIGRTMIN

/******************
 * ERROR HANDLING *
 ******************/

typedef enum {
    TRACEE_SUCCESS,
    TRACEE_ERR_NO_SUCH_PID,
    TRACEE_ERR_PARAM,
    TRACEE_ERR_PERM,
    TRACEE_ERR_UNREACHABLE,
    TRACEE_ERR_EXITED,
    TRACEE_ERR_OTHER,
    TRACEE_ERR_ALREADY_TRACKED,
} tracee_error_t;

static const char* const tracee_error_str[TRACEE_ERR_ALREADY_TRACKED + 1] = {
    "TRACEE_SUCCESS",
    "TRACEE_ERR_NO_SUCH_PID",
    "TRACEE_ERR_PARAM",
    "TRACEE_ERR_PERM",
    "TRACEE_ERR_UNREACHABLE",
    "TRACEE_ERR_EXITED",
    "TRACEE_ERR_OTHER",
    "TRACEE_ERR_ALREADY_TRACKED"
};

/*********************
 * TRACKING INTERFACE *
 **********************/

/**
 * @brief For the parent to attach and traces its child for ptrace.
 *
 * @param pid of the child
 * @return EXIT_SUCCESS or EXIT_FAILURE
 */
int track_process(pid_t pid);

/**
 * @brief For the child to be traced by its parent.
 *        Set the tracer as pid and raise(SIGSTOP) to wait the tracer to be attached.
 * 
 * @param pid of the tracer/parent
 */
void prepare_to_be_tracked(pid_t pid);

#endif /* TRACEE_H */

