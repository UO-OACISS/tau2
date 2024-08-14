# SYSCALL wrapper with ptrace

Syscall wrapper that uses ptrace to trace with tau each syscall called.
Configure TAU with the `-syscall` option and run with `tau_exec -syscall`

## Limitations

The syscall wrapper does not work with tracing.


## Add an architecture

See /usr/include/asm/uninstd.h to get the `__NR_` macros corresponding to the id and name of each syscall.

### If PTRACE_GETREGS is present

Check `/usr/include/sys/user.h` and `/usr/include/asm/ptrace.h` for the structure of the registers get by ptrace when doing `PTRACE_GETREGS`.

`man syscall` (See part "Architecture calling conventions") to get the system call number.

See example in [scalls_ppc.c](./scalls_ppc.c)

### If PTRACE_GETREGS is NOT present

Then we have to use PTRACE_GETREGSET.
See example in [scalls_aarch64.c](./scalls_aarch64.c)

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
- if it's because the brand new child just starts and stops, then the parent and the main child initiate a synchronization of their TAU task ids number. When the synchronization is done, then the child can be start wih the correct task id.
- if a child is exiting, then the parent detaches it and stops to track it
- for other reasons: the parent just relaunches the child and continues the tracking

The tracking will be stopped if the parent will receive the signal `SIGRTMIN`, if the main child exits or if any error occurs for the parent.

### scalls.c

Used to get the syscall name depending on the syscall number.
One file per architecture since each architecture has its own syscalls.


## Brief description of the algorithm

### At initialisation, before fork()

Creation and initalisation of all shared variables. (`taupreload_init()`)

### Main Child (Main process)

- Initialisation of TAU. (`taupreload_main()`)
- Attaching process to the parent (`prepare_to_be_tracked()`)
    - Set the permission for the parent to track the child
    - Create the task creator
    - Assign a TAU task id for the parent
    - Stop the process and wait for the parent to attach it
    - Restart the task creator if needed
- Run the program
- Send signal `SIG_STOP_PTRACE` to stop the tracking
- Wait for the parent to dump its TAU profile files
- End TAU (dump profile files, etc)

### Task creator

Its only goal is to do `TAU_CREATE_TASK()` when needed and to update the shared number of task ids between the child and the parent.
Its syscalls are not tracked.
It uses a `pthread_mutex` and a `pthread_cond` shared with the parent for the synchronization.

### Parent (Syscall tracker)

(Here, when I say "(re)start a process", I mean "ask the process to run until next syscall")

#### Attach the child and start it

- Initialisation of TAU. (`taupreload_main()`)
- Attach the child (`tracee_seize()`)
    - Wait for the child to stop
    - Use `PTRACE_SEIZE` to attach the child. The options used allow the parent to track all threads of the child.
- Set the correct TAU task id for the child and start it 
- Enter the main loop

#### Main loop

The loop is in the `tracee_track_syscall()` function and loops until `ending_tracking` is set on

- Wait for any child thread to stop (`tracee_wait_for_child()`)
- Check the reason for the stop
    - `WAIT_EXITED`: the thread has exited
    - `WAIT_STOPPED`: the thread was stopped because of a `SIGSTOP` signal
    - `WAIT_STOPPED_OTHER`: the thread was stopped because it receives another signal
    - `WAIT_STOPPED_NEW_CHILD`: the thread has just been created and hasn't started yet
    - `WAIT_SYSCALL` (and all `WAIT_SYSCALL_FOO`): the thread enters or exits a syscall
- Do the [New Child Routine](#new-child-routine)
- Act depending on the reason of the stop:
    - if `WAIT_EXITED`, stop the tracking of the thread
    - if `WAIT_STOPPED`, restart the thread
    - if `WAIT_STOPPED_OTHER`, restart the thread and inject the signal it received
    - if `WAIT_STOPPED_NEW_CHILD`, do nothing. (See [New Child Routine](#new-child-routine))
    - if `WAIT_SYSCALL`, 
        - if the child is exiting the syscall, stop the timer and restart the thread
        - else, get the syscall name, start a TAU timer and restart the thread
- Continue the loop

#### New Child Routine

- If the task creator is not working
    - Take the oldest child not yet started (`get_waiting_new_child_tt()`)
        - If there is no such child, stop the routine
        - If the child hasn't asks the task creator to work 
            - Ask the task creator to wake up and work
        - Else
            - Set the correct task id for the child, start the TAU top timer and start the thread (`tracee_start_tracking_tt()`)

#### Ending the tracking

If the main loop is stopped (`ending_tracking = 1`).

- End TAU (dump profile files, etc)
- Notify the main child and task creator that they can safely exit
- For all threads:
    - If not stopped, interrupt the thread with `PTRACE_INTERRUPT`
    - Detach the thread with `PTRACE_DETACH` (It also restarts the thread)
- Wait for child to exit
- Free shared variables
- Exit
