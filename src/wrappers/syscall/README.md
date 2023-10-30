# SYSCALL wrapper with ptrace

Syscall wrapper that uses ptrace to trace with tau each syscall called.

WARNING: the current version only track the syscall of the main thread.
You can unset the flag NO_TRACECLONE in tracee.c to try with multiple threads, but it may result in a blocking process when profiling.

## Add an architecture

See /usr/include/asm/uninstd.h to get the `__NR_` macros corresponding to the id and name of each syscall.

### If PTRACE_GETREGS is present

Check `/usr/include/sys/user.h` and `/usr/include/asm/ptrace.h` for the structure of the registers get by ptrace when doing `PTRACE_GETREGS`.

`man syscall` (See part "Architecture calling conventions") to get the system call number.

See example in `scalls_ppc.c`

### If PTRACE_GETREGS is NOT present

Then we have to use PTRACE_GETREGSET.
See example in scalls_aarch64.c

## Organisation

### ptrace_syscall.c

A fork() occurs at the start of the program.
The child will wait to be attached to its parent before starting the main.

(This file is just a modified version of `../taupreload/taupreload.c`)

### tracee.c

Main file.

The child should only use `prepare_to_be_tracked(getppid())` while the parent uses `track_process(child_pid)`.
The parent attaches the child and then enters in a loop (`tracee_track_syscall`) in which it waits for its child or one of the child's threads to be stopped (`wait_for_pid`).
The child can be stopped for several reasons (see enum `tracee_wait_t`).

Depending on the reason, the parent acts differently:
- if it's because of a SYSCALL (`WAIT_SYSCALL`), then it can be the enter of the exit of the syscall. So the parent starts or stops a TAU timer for this syscall.
- if it's because of a clone syscall, then the child is creating a thread, so the parent registers the pid of this new thread to track it
- if it's because the brand new child just starts and stops, then the parent starts a timer for the duration of the thread
- if a child is exiting, then the parent stops to track it and it stops the timers of this thread
- for other reasons: the parent just relaunches the child and continues the tracking

The tracking will be stopped if the parent will receive the signal `SIGRTMIN` or if the main child exits.

### scalls.c

Used to get the syscall name depending on the syscall number.
One file per architecture since each architecture has its own syscalls.


