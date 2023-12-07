#ifdef __aarch64__

#include "scalls.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/user.h>

#include <elf.h> // See /usr/include/elf.h
#include <sys/uio.h>

/*****************
 * NAME HANDLING *
 *****************/

#define SCALL_SIZE 1024

static const char *__scalls[SCALL_SIZE];

const char *get_syscall_name(int id)
{
    if ((SCALL_SIZE <= id) || (id < 0))
    {
        return "UNKNOWN";
    }

    return __scalls[id];
}

// Depends on the architecture!

void scalls_init(void)
{
    int i;

    for (i = 0; i < SCALL_SIZE; i++)
    {
        __scalls[i] = "UNKNOWN";
    }

    __scalls[__NR_io_setup] = "io_setup";
    __scalls[__NR_io_destroy] = "io_destroy";
    __scalls[__NR_io_submit] = "io_submit";
    __scalls[__NR_io_cancel] = "io_cancel";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_io_getevents] = "io_getevents";
#endif
    __scalls[__NR_setxattr] = "setxattr";
    __scalls[__NR_lsetxattr] = "lsetxattr";
    __scalls[__NR_fsetxattr] = "fsetxattr";
    __scalls[__NR_getxattr] = "getxattr";
    __scalls[__NR_lgetxattr] = "lgetxattr";
    __scalls[__NR_fgetxattr] = "fgetxattr";
    __scalls[__NR_listxattr] = "listxattr";
    __scalls[__NR_llistxattr] = "llistxattr";
    __scalls[__NR_flistxattr] = "flistxattr";
    __scalls[__NR_removexattr] = "removexattr";
    __scalls[__NR_lremovexattr] = "lremovexattr";
    __scalls[__NR_fremovexattr] = "fremovexattr";
    __scalls[__NR_getcwd] = "getcwd";
    __scalls[__NR_lookup_dcookie] = "lookup_dcookie";
    __scalls[__NR_eventfd2] = "eventfd2";
    __scalls[__NR_epoll_create1] = "epoll_create1";
    __scalls[__NR_epoll_ctl] = "epoll_ctl";
    __scalls[__NR_epoll_pwait] = "epoll_pwait";
    __scalls[__NR_dup] = "dup";
    __scalls[__NR_dup3] = "dup3";
    __scalls[__NR3264_fcntl] = "fcntl";
    __scalls[__NR_inotify_init1] = "inotify_init1";
    __scalls[__NR_inotify_add_watch] = "inotify_add_watch";
    __scalls[__NR_inotify_rm_watch] = "inotify_rm_watch";
    __scalls[__NR_ioctl] = "ioctl";
    __scalls[__NR_ioprio_set] = "ioprio_set";
    __scalls[__NR_ioprio_get] = "ioprio_get";
    __scalls[__NR_flock] = "flock";
    __scalls[__NR_mknodat] = "mknodat";
    __scalls[__NR_mkdirat] = "mkdirat";
    __scalls[__NR_unlinkat] = "unlinkat";
    __scalls[__NR_symlinkat] = "symlinkat";
    __scalls[__NR_linkat] = "linkat";
#ifdef __ARCH_WANT_RENAMEAT
    __scalls[__NR_renameat] = "renameat";
#endif
    __scalls[__NR_umount2] = "umount2";
    __scalls[__NR_mount] = "mount";
    __scalls[__NR_pivot_root] = "pivot_root";
    __scalls[__NR_nfsservctl] = "nfsservctl";
    __scalls[__NR3264_statfs] = "statfs";
    __scalls[__NR3264_fstatfs] = "fstatfs";
    __scalls[__NR3264_truncate] = "truncate";
    __scalls[__NR3264_ftruncate] = "ftruncate";
    __scalls[__NR_fallocate] = "fallocate";
    __scalls[__NR_faccessat] = "faccessat";
    __scalls[__NR_chdir] = "chdir";
    __scalls[__NR_fchdir] = "fchdir";
    __scalls[__NR_chroot] = "chroot";
    __scalls[__NR_fchmod] = "fchmod";
    __scalls[__NR_fchmodat] = "fchmodat";
    __scalls[__NR_fchownat] = "fchownat";
    __scalls[__NR_fchown] = "fchown";
    __scalls[__NR_openat] = "openat";
    __scalls[__NR_close] = "close";
    __scalls[__NR_vhangup] = "vhangup";
    __scalls[__NR_pipe2] = "pipe2";
    __scalls[__NR_quotactl] = "quotactl";
    __scalls[__NR_getdents64] = "getdents64";
    __scalls[__NR3264_lseek] = "lseek";
    __scalls[__NR_read] = "read";
    __scalls[__NR_write] = "write";
    __scalls[__NR_readv] = "readv";
    __scalls[__NR_writev] = "writev";
    __scalls[__NR_pread64] = "pread64";
    __scalls[__NR_pwrite64] = "pwrite64";
    __scalls[__NR_preadv] = "preadv";
    __scalls[__NR_pwritev] = "pwritev";
    __scalls[__NR3264_sendfile] = "sendfile";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_pselect6] = "pselect6";
    __scalls[__NR_ppoll] = "ppoll";
#endif
    __scalls[__NR_signalfd4] = "signalfd4";
    __scalls[__NR_vmsplice] = "vmsplice";
    __scalls[__NR_splice] = "splice";
    __scalls[__NR_tee] = "tee";
    __scalls[__NR_readlinkat] = "readlinkat";
#if defined(__ARCH_WANT_NEW_STAT) || defined(__ARCH_WANT_STAT64)
    __scalls[__NR3264_fstatat] = "fstatat";
    __scalls[__NR3264_fstat] = "fstat";
#endif
    __scalls[__NR_sync] = "sync";
    __scalls[__NR_fsync] = "fsync";
    __scalls[__NR_fdatasync] = "fdatasync";
#ifdef __ARCH_WANT_SYNC_FILE_RANGE2
    __scalls[__NR_sync_file_range2] = "sync_file_range2";
#else
    __scalls[__NR_sync_file_range] = "sync_file_range";
#endif
    __scalls[__NR_timerfd_create] = "timerfd_create";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_timerfd_settime] = "timerfd_settime";
    __scalls[__NR_timerfd_gettime] = "timerfd_gettime";
#endif
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_utimensat] = "utimensat";
#endif
    __scalls[__NR_acct] = "acct";
    __scalls[__NR_capget] = "capget";
    __scalls[__NR_capset] = "capset";
    __scalls[__NR_personality] = "personality";
    __scalls[__NR_exit] = "exit";
    __scalls[__NR_exit_group] = "exit_group";
    __scalls[__NR_waitid] = "waitid";
    __scalls[__NR_set_tid_address] = "set_tid_address";
    __scalls[__NR_unshare] = "unshare";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_futex] = "futex";
#endif
    __scalls[__NR_set_robust_list] = "set_robust_list";
    __scalls[__NR_get_robust_list] = "get_robust_list";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_nanosleep] = "nanosleep";
#endif
    __scalls[__NR_getitimer] = "getitimer";
    __scalls[__NR_setitimer] = "setitimer";
    __scalls[__NR_kexec_load] = "kexec_load";
    __scalls[__NR_init_module] = "init_module";
    __scalls[__NR_delete_module] = "delete_module";
    __scalls[__NR_timer_create] = "timer_create";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_timer_gettime] = "timer_gettime";
#endif
    __scalls[__NR_timer_getoverrun] = "timer_getoverrun";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_timer_settime] = "timer_settime";
#endif
    __scalls[__NR_timer_delete] = "timer_delete";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_clock_settime] = "clock_settime";
    __scalls[__NR_clock_gettime] = "clock_gettime";
    __scalls[__NR_clock_getres] = "clock_getres";
    __scalls[__NR_clock_nanosleep] = "clock_nanosleep";
#endif
    __scalls[__NR_syslog] = "syslog";
    __scalls[__NR_ptrace] = "ptrace";
    __scalls[__NR_sched_setparam] = "sched_setparam";
    __scalls[__NR_sched_setscheduler] = "sched_setscheduler";
    __scalls[__NR_sched_getscheduler] = "sched_getscheduler";
    __scalls[__NR_sched_getparam] = "sched_getparam";
    __scalls[__NR_sched_setaffinity] = "sched_setaffinity";
    __scalls[__NR_sched_getaffinity] = "sched_getaffinity";
    __scalls[__NR_sched_yield] = "sched_yield";
    __scalls[__NR_sched_get_priority_max] = "sched_get_priority_max";
    __scalls[__NR_sched_get_priority_min] = "sched_get_priority_min";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_sched_rr_get_interval] = "sched_rr_get_interval";
#endif
    __scalls[__NR_restart_syscall] = "restart_syscall";
    __scalls[__NR_kill] = "kill";
    __scalls[__NR_tkill] = "tkill";
    __scalls[__NR_tgkill] = "tgkill";
    __scalls[__NR_sigaltstack] = "sigaltstack";
    __scalls[__NR_rt_sigsuspend] = "rt_sigsuspend";
    __scalls[__NR_rt_sigaction] = "rt_sigaction";
    __scalls[__NR_rt_sigprocmask] = "rt_sigprocmask";
    __scalls[__NR_rt_sigpending] = "rt_sigpending";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_rt_sigtimedwait] = "rt_sigtimedwait";
#endif
    __scalls[__NR_rt_sigqueueinfo] = "rt_sigqueueinfo";
    __scalls[__NR_rt_sigreturn] = "rt_sigreturn";
    __scalls[__NR_setpriority] = "setpriority";
    __scalls[__NR_getpriority] = "getpriority";
    __scalls[__NR_reboot] = "reboot";
    __scalls[__NR_setregid] = "setregid";
    __scalls[__NR_setgid] = "setgid";
    __scalls[__NR_setreuid] = "setreuid";
    __scalls[__NR_setuid] = "setuid";
    __scalls[__NR_setresuid] = "setresuid";
    __scalls[__NR_getresuid] = "getresuid";
    __scalls[__NR_setresgid] = "setresgid";
    __scalls[__NR_getresgid] = "getresgid";
    __scalls[__NR_setfsuid] = "setfsuid";
    __scalls[__NR_setfsgid] = "setfsgid";
    __scalls[__NR_times] = "times";
    __scalls[__NR_setpgid] = "setpgid";
    __scalls[__NR_getpgid] = "getpgid";
    __scalls[__NR_getsid] = "getsid";
    __scalls[__NR_setsid] = "setsid";
    __scalls[__NR_getgroups] = "getgroups";
    __scalls[__NR_setgroups] = "setgroups";
    __scalls[__NR_uname] = "uname";
    __scalls[__NR_sethostname] = "sethostname";
    __scalls[__NR_setdomainname] = "setdomainname";
#ifdef __ARCH_WANT_SET_GET_RLIMIT
    __scalls[__NR_getrlimit] = "getrlimit";
    __scalls[__NR_setrlimit] = "setrlimit";
#endif
    __scalls[__NR_getrusage] = "getrusage";
    __scalls[__NR_umask] = "umask";
    __scalls[__NR_prctl] = "prctl";
    __scalls[__NR_getcpu] = "getcpu";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_gettimeofday] = "gettimeofday";
    __scalls[__NR_settimeofday] = "settimeofday";
    __scalls[__NR_adjtimex] = "adjtimex";
#endif
    __scalls[__NR_getpid] = "getpid";
    __scalls[__NR_getppid] = "getppid";
    __scalls[__NR_getuid] = "getuid";
    __scalls[__NR_geteuid] = "geteuid";
    __scalls[__NR_getgid] = "getgid";
    __scalls[__NR_getegid] = "getegid";
    __scalls[__NR_gettid] = "gettid";
    __scalls[__NR_sysinfo] = "sysinfo";
    __scalls[__NR_mq_open] = "mq_open";
    __scalls[__NR_mq_unlink] = "mq_unlink";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_mq_timedsend] = "mq_timedsend";
    __scalls[__NR_mq_timedreceive] = "mq_timedreceive";
#endif
    __scalls[__NR_mq_notify] = "mq_notify";
    __scalls[__NR_mq_getsetattr] = "mq_getsetattr";
    __scalls[__NR_msgget] = "msgget";
    __scalls[__NR_msgctl] = "msgctl";
    __scalls[__NR_msgrcv] = "msgrcv";
    __scalls[__NR_msgsnd] = "msgsnd";
    __scalls[__NR_semget] = "semget";
    __scalls[__NR_semctl] = "semctl";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_semtimedop] = "semtimedop";
#endif
    __scalls[__NR_semop] = "semop";
    __scalls[__NR_shmget] = "shmget";
    __scalls[__NR_shmctl] = "shmctl";
    __scalls[__NR_shmat] = "shmat";
    __scalls[__NR_shmdt] = "shmdt";
    __scalls[__NR_socket] = "socket";
    __scalls[__NR_socketpair] = "socketpair";
    __scalls[__NR_bind] = "bind";
    __scalls[__NR_listen] = "listen";
    __scalls[__NR_accept] = "accept";
    __scalls[__NR_connect] = "connect";
    __scalls[__NR_getsockname] = "getsockname";
    __scalls[__NR_getpeername] = "getpeername";
    __scalls[__NR_sendto] = "sendto";
    __scalls[__NR_recvfrom] = "recvfrom";
    __scalls[__NR_setsockopt] = "setsockopt";
    __scalls[__NR_getsockopt] = "getsockopt";
    __scalls[__NR_shutdown] = "shutdown";
    __scalls[__NR_sendmsg] = "sendmsg";
    __scalls[__NR_recvmsg] = "recvmsg";
    __scalls[__NR_readahead] = "readahead";
    __scalls[__NR_brk] = "brk";
    __scalls[__NR_munmap] = "munmap";
    __scalls[__NR_mremap] = "mremap";
    __scalls[__NR_add_key] = "add_key";
    __scalls[__NR_request_key] = "request_key";
    __scalls[__NR_keyctl] = "keyctl";
    __scalls[__NR_clone] = "clone";
    __scalls[__NR_execve] = "execve";
    __scalls[__NR3264_mmap] = "mmap";
    __scalls[__NR3264_fadvise64] = "fadvise64";
#ifndef __ARCH_NOMMU
    __scalls[__NR_swapon] = "swapon";
    __scalls[__NR_swapoff] = "swapoff";
    __scalls[__NR_mprotect] = "mprotect";
    __scalls[__NR_msync] = "msync";
    __scalls[__NR_mlock] = "mlock";
    __scalls[__NR_munlock] = "munlock";
    __scalls[__NR_mlockall] = "mlockall";
    __scalls[__NR_munlockall] = "munlockall";
    __scalls[__NR_mincore] = "mincore";
    __scalls[__NR_madvise] = "madvise";
    __scalls[__NR_remap_file_pages] = "remap_file_pages";
    __scalls[__NR_mbind] = "mbind";
    __scalls[__NR_get_mempolicy] = "get_mempolicy";
    __scalls[__NR_set_mempolicy] = "set_mempolicy";
    __scalls[__NR_migrate_pages] = "migrate_pages";
    __scalls[__NR_move_pages] = "move_pages";
#endif
    __scalls[__NR_rt_tgsigqueueinfo] = "rt_tgsigqueueinfo";
    __scalls[__NR_perf_event_open] = "perf_event_open";
    __scalls[__NR_accept4] = "accept4";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_recvmmsg] = "recvmmsg";
#endif
    __scalls[__NR_arch_specific_syscall] = "arch_specific_syscall";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_wait4] = "wait4";
#endif
    __scalls[__NR_prlimit64] = "prlimit64";
    __scalls[__NR_fanotify_init] = "fanotify_init";
    __scalls[__NR_fanotify_mark] = "fanotify_mark";
    __scalls[__NR_name_to_handle_at] = "name_to_handle_at";
    __scalls[__NR_open_by_handle_at] = "open_by_handle_at";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_clock_adjtime] = "clock_adjtime";
#endif
    __scalls[__NR_syncfs] = "syncfs";
    __scalls[__NR_setns] = "setns";
    __scalls[__NR_sendmmsg] = "sendmmsg";
    __scalls[__NR_process_vm_readv] = "process_vm_readv";
    __scalls[__NR_process_vm_writev] = "process_vm_writev";
    __scalls[__NR_kcmp] = "kcmp";
    __scalls[__NR_finit_module] = "finit_module";
    __scalls[__NR_sched_setattr] = "sched_setattr";
    __scalls[__NR_sched_getattr] = "sched_getattr";
    __scalls[__NR_renameat2] = "renameat2";
    __scalls[__NR_seccomp] = "seccomp";
    __scalls[__NR_getrandom] = "getrandom";
    __scalls[__NR_memfd_create] = "memfd_create";
    __scalls[__NR_bpf] = "bpf";
    __scalls[__NR_execveat] = "execveat";
    __scalls[__NR_userfaultfd] = "userfaultfd";
    __scalls[__NR_membarrier] = "membarrier";
    __scalls[__NR_mlock2] = "mlock2";
    __scalls[__NR_copy_file_range] = "copy_file_range";
    __scalls[__NR_preadv2] = "preadv2";
    __scalls[__NR_pwritev2] = "pwritev2";
    __scalls[__NR_pkey_mprotect] = "pkey_mprotect";
    __scalls[__NR_pkey_alloc] = "pkey_alloc";
    __scalls[__NR_pkey_free] = "pkey_free";
    __scalls[__NR_statx] = "statx";
#if defined(__ARCH_WANT_TIME32_SYSCALLS) || __BITS_PER_LONG != 32
    __scalls[__NR_io_pgetevents] = "io_pgetevents";
#endif
    __scalls[__NR_rseq] = "rseq";
    __scalls[__NR_kexec_file_load] = "kexec_file_load";
#if __BITS_PER_LONG == 32
    __scalls[__NR_clock_gettime64] = "clock_gettime64";
    __scalls[__NR_clock_settime64] = "clock_settime64";
    __scalls[__NR_clock_adjtime64] = "clock_adjtime64";
    __scalls[__NR_clock_getres_time64] = "clock_getres_time64";
    __scalls[__NR_clock_nanosleep_time64] = "clock_nanosleep_time64";
    __scalls[__NR_timer_gettime64] = "timer_gettime64";
    __scalls[__NR_timer_settime64] = "timer_settime64";
    __scalls[__NR_timerfd_gettime64] = "timerfd_gettime64";
    __scalls[__NR_timerfd_settime64] = "timerfd_settime64";
    __scalls[__NR_utimensat_time64] = "utimensat_time64";
    __scalls[__NR_pselect6_time64] = "pselect6_time64";
    __scalls[__NR_ppoll_time64] = "ppoll_time64";
    __scalls[__NR_io_pgetevents_time64] = "io_pgetevents_time64";
    __scalls[__NR_recvmmsg_time64] = "recvmmsg_time64";
    __scalls[__NR_mq_timedsend_time64] = "mq_timedsend_time64";
    __scalls[__NR_mq_timedreceive_time64] = "mq_timedreceive_time64";
    __scalls[__NR_semtimedop_time64] = "semtimedop_time64";
    __scalls[__NR_rt_sigtimedwait_time64] = "rt_sigtimedwait_time64";
    __scalls[__NR_futex_time64] = "futex_time64";
    __scalls[__NR_sched_rr_get_interval_time64] = "sched_rr_get_interval_time64";
#endif
    __scalls[__NR_pidfd_send_signal] = "pidfd_send_signal";
    __scalls[__NR_io_uring_setup] = "io_uring_setup";
    __scalls[__NR_io_uring_enter] = "io_uring_enter";
    __scalls[__NR_io_uring_register] = "io_uring_register";
    __scalls[__NR_open_tree] = "open_tree";
    __scalls[__NR_move_mount] = "move_mount";
    __scalls[__NR_fsopen] = "fsopen";
    __scalls[__NR_fsconfig] = "fsconfig";
    __scalls[__NR_fsmount] = "fsmount";
    __scalls[__NR_fspick] = "fspick";
    __scalls[__NR_pidfd_open] = "pidfd_open";
#ifdef __ARCH_WANT_SYS_CLONE3
    __scalls[__NR_clone3] = "clone3";
#endif
    __scalls[__NR_syscalls] = "syscalls";
}

/*
struct user_regs_struct
{
  unsigned long long regs[31];
  unsigned long long sp;
  unsigned long long pc;
  unsigned long long pstate;
};

*/

/*
       struct iovec {
           void   *iov_base;  // Starting address
           size_t  iov_len;   // Size of the memory pointed to by iov_base.
       };
*/

static void printf_regs(struct user_regs_struct regs)
{
    for (int i = 0; i < 32; i++)
    {
        printf("regs[%d] : %lld\n", i, regs.regs[i]);
    }

    printf("sp : %lld\n", regs.sp);
    printf("pc : %lld\n", regs.pc);
    printf("pstate : %lld\n", regs.pstate);
}

int get_syscall_id(pid_t pid)
{
    // struct user_regs_struct regs;
    int i = -1;
    struct iovec iovector;
    iovector.iov_base = &i;
    iovector.iov_len = sizeof(int);

    // Macro found in elf.h : "ARM system call number"
    if (ptrace(PTRACE_GETREGSET, pid, NT_ARM_SYSTEM_CALL, &iovector) < 0)
    {
        perror("ptrace (getregset)");
        return -1;
    }

    return i;
}

#endif
