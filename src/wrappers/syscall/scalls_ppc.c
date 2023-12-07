#ifdef __PPC__

#include "scalls.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/user.h>

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
    // __scalls[__NR_exit] = "exit"; // see "/usr/include/asm/unistd.h"
    __scalls[__NR_restart_syscall] = "restart_syscall";
    __scalls[__NR_exit] = "exit";
    __scalls[__NR_fork] = "fork";
    __scalls[__NR_read] = "read";
    __scalls[__NR_write] = "write";
    __scalls[__NR_open] = "open";
    __scalls[__NR_close] = "close";
    __scalls[__NR_waitpid] = "waitpid";
    __scalls[__NR_creat] = "creat";
    __scalls[__NR_link] = "link";
    __scalls[__NR_unlink] = "unlink";
    __scalls[__NR_execve] = "execve";
    __scalls[__NR_chdir] = "chdir";
    __scalls[__NR_time] = "time";
    __scalls[__NR_mknod] = "mknod";
    __scalls[__NR_chmod] = "chmod";
    __scalls[__NR_lchown] = "lchown";
    __scalls[__NR_break] = "break";
    __scalls[__NR_oldstat] = "oldstat";
    __scalls[__NR_lseek] = "lseek";
    __scalls[__NR_getpid] = "getpid";
    __scalls[__NR_mount] = "mount";
    __scalls[__NR_umount] = "umount";
    __scalls[__NR_setuid] = "setuid";
    __scalls[__NR_getuid] = "getuid";
    __scalls[__NR_stime] = "stime";
    __scalls[__NR_ptrace] = "ptrace";
    __scalls[__NR_alarm] = "alarm";
    __scalls[__NR_oldfstat] = "oldfstat";
    __scalls[__NR_pause] = "pause";
    __scalls[__NR_utime] = "utime";
    __scalls[__NR_stty] = "stty";
    __scalls[__NR_gtty] = "gtty";
    __scalls[__NR_access] = "access";
    __scalls[__NR_nice] = "nice";
    __scalls[__NR_ftime] = "ftime";
    __scalls[__NR_sync] = "sync";
    __scalls[__NR_kill] = "kill";
    __scalls[__NR_rename] = "rename";
    __scalls[__NR_mkdir] = "mkdir";
    __scalls[__NR_rmdir] = "rmdir";
    __scalls[__NR_dup] = "dup";
    __scalls[__NR_pipe] = "pipe";
    __scalls[__NR_times] = "times";
    __scalls[__NR_prof] = "prof";
    __scalls[__NR_brk] = "brk";
    __scalls[__NR_setgid] = "setgid";
    __scalls[__NR_getgid] = "getgid";
    __scalls[__NR_signal] = "signal";
    __scalls[__NR_geteuid] = "geteuid";
    __scalls[__NR_getegid] = "getegid";
    __scalls[__NR_acct] = "acct";
    __scalls[__NR_umount2] = "umount2";
    __scalls[__NR_lock] = "lock";
    __scalls[__NR_ioctl] = "ioctl";
    __scalls[__NR_fcntl] = "fcntl";
    __scalls[__NR_mpx] = "mpx";
    __scalls[__NR_setpgid] = "setpgid";
    __scalls[__NR_ulimit] = "ulimit";
    __scalls[__NR_oldolduname] = "oldolduname";
    __scalls[__NR_umask] = "umask";
    __scalls[__NR_chroot] = "chroot";
    __scalls[__NR_ustat] = "ustat";
    __scalls[__NR_dup2] = "dup2";
    __scalls[__NR_getppid] = "getppid";
    __scalls[__NR_getpgrp] = "getpgrp";
    __scalls[__NR_setsid] = "setsid";
    __scalls[__NR_sigaction] = "sigaction";
    __scalls[__NR_sgetmask] = "sgetmask";
    __scalls[__NR_ssetmask] = "ssetmask";
    __scalls[__NR_setreuid] = "setreuid";
    __scalls[__NR_setregid] = "setregid";
    __scalls[__NR_sigsuspend] = "sigsuspend";
    __scalls[__NR_sigpending] = "sigpending";
    __scalls[__NR_sethostname] = "sethostname";
    __scalls[__NR_setrlimit] = "setrlimit";
    __scalls[__NR_getrlimit] = "getrlimit";
    __scalls[__NR_getrusage] = "getrusage";
    __scalls[__NR_gettimeofday] = "gettimeofday";
    __scalls[__NR_settimeofday] = "settimeofday";
    __scalls[__NR_getgroups] = "getgroups";
    __scalls[__NR_setgroups] = "setgroups";
    __scalls[__NR_select] = "select";
    __scalls[__NR_symlink] = "symlink";
    __scalls[__NR_oldlstat] = "oldlstat";
    __scalls[__NR_readlink] = "readlink";
    __scalls[__NR_uselib] = "uselib";
    __scalls[__NR_swapon] = "swapon";
    __scalls[__NR_reboot] = "reboot";
    __scalls[__NR_readdir] = "readdir";
    __scalls[__NR_mmap] = "mmap";
    __scalls[__NR_munmap] = "munmap";
    __scalls[__NR_truncate] = "truncate";
    __scalls[__NR_ftruncate] = "ftruncate";
    __scalls[__NR_fchmod] = "fchmod";
    __scalls[__NR_fchown] = "fchown";
    __scalls[__NR_getpriority] = "getpriority";
    __scalls[__NR_setpriority] = "setpriority";
    __scalls[__NR_profil] = "profil";
    __scalls[__NR_statfs] = "statfs";
    __scalls[__NR_fstatfs] = "fstatfs";
    __scalls[__NR_ioperm] = "ioperm";
    __scalls[__NR_socketcall] = "socketcall";
    __scalls[__NR_syslog] = "syslog";
    __scalls[__NR_setitimer] = "setitimer";
    __scalls[__NR_getitimer] = "getitimer";
    __scalls[__NR_stat] = "stat";
    __scalls[__NR_lstat] = "lstat";
    __scalls[__NR_fstat] = "fstat";
    __scalls[__NR_olduname] = "olduname";
    __scalls[__NR_iopl] = "iopl";
    __scalls[__NR_vhangup] = "vhangup";
    __scalls[__NR_idle] = "idle";
    __scalls[__NR_vm86] = "vm86";
    __scalls[__NR_wait4] = "wait4";
    __scalls[__NR_swapoff] = "swapoff";
    __scalls[__NR_sysinfo] = "sysinfo";
    __scalls[__NR_ipc] = "ipc";
    __scalls[__NR_fsync] = "fsync";
    __scalls[__NR_sigreturn] = "sigreturn";
    __scalls[__NR_clone] = "clone";
    __scalls[__NR_setdomainname] = "setdomainname";
    __scalls[__NR_uname] = "uname";
    __scalls[__NR_modify_ldt] = "modify_ldt";
    __scalls[__NR_adjtimex] = "adjtimex";
    __scalls[__NR_mprotect] = "mprotect";
    __scalls[__NR_sigprocmask] = "sigprocmask";
    __scalls[__NR_create_module] = "create_module";
    __scalls[__NR_init_module] = "init_module";
    __scalls[__NR_delete_module] = "delete_module";
    __scalls[__NR_get_kernel_syms] = "get_kernel_syms";
    __scalls[__NR_quotactl] = "quotactl";
    __scalls[__NR_getpgid] = "getpgid";
    __scalls[__NR_fchdir] = "fchdir";
    __scalls[__NR_bdflush] = "bdflush";
    __scalls[__NR_sysfs] = "sysfs";
    __scalls[__NR_personality] = "personality";
    __scalls[__NR_afs_syscall] = "afs_syscall";
    __scalls[__NR_setfsuid] = "setfsuid";
    __scalls[__NR_setfsgid] = "setfsgid";
    __scalls[__NR__llseek] = "_llseek";
    __scalls[__NR_getdents] = "getdents";
    __scalls[__NR__newselect] = "_newselect";
    __scalls[__NR_flock] = "flock";
    __scalls[__NR_msync] = "msync";
    __scalls[__NR_readv] = "readv";
    __scalls[__NR_writev] = "writev";
    __scalls[__NR_getsid] = "getsid";
    __scalls[__NR_fdatasync] = "fdatasync";
    __scalls[__NR__sysctl] = "_sysctl";
    __scalls[__NR_mlock] = "mlock";
    __scalls[__NR_munlock] = "munlock";
    __scalls[__NR_mlockall] = "mlockall";
    __scalls[__NR_munlockall] = "munlockall";
    __scalls[__NR_sched_setparam] = "sched_setparam";
    __scalls[__NR_sched_getparam] = "sched_getparam";
    __scalls[__NR_sched_setscheduler] = "sched_setscheduler";
    __scalls[__NR_sched_getscheduler] = "sched_getscheduler";
    __scalls[__NR_sched_yield] = "sched_yield";
    __scalls[__NR_sched_get_priority_max] = "sched_get_priority_max";
    __scalls[__NR_sched_get_priority_min] = "sched_get_priority_min";
    __scalls[__NR_sched_rr_get_interval] = "sched_rr_get_interval";
    __scalls[__NR_nanosleep] = "nanosleep";
    __scalls[__NR_mremap] = "mremap";
    __scalls[__NR_setresuid] = "setresuid";
    __scalls[__NR_getresuid] = "getresuid";
    __scalls[__NR_query_module] = "query_module";
    __scalls[__NR_poll] = "poll";
    __scalls[__NR_nfsservctl] = "nfsservctl";
    __scalls[__NR_setresgid] = "setresgid";
    __scalls[__NR_getresgid] = "getresgid";
    __scalls[__NR_prctl] = "prctl";
    __scalls[__NR_rt_sigreturn] = "rt_sigreturn";
    __scalls[__NR_rt_sigaction] = "rt_sigaction";
    __scalls[__NR_rt_sigprocmask] = "rt_sigprocmask";
    __scalls[__NR_rt_sigpending] = "rt_sigpending";
    __scalls[__NR_rt_sigtimedwait] = "rt_sigtimedwait";
    __scalls[__NR_rt_sigqueueinfo] = "rt_sigqueueinfo";
    __scalls[__NR_rt_sigsuspend] = "rt_sigsuspend";
    __scalls[__NR_pread64] = "pread64";
    __scalls[__NR_pwrite64] = "pwrite64";
    __scalls[__NR_chown] = "chown";
    __scalls[__NR_getcwd] = "getcwd";
    __scalls[__NR_capget] = "capget";
    __scalls[__NR_capset] = "capset";
    __scalls[__NR_sigaltstack] = "sigaltstack";
    __scalls[__NR_sendfile] = "sendfile";
    __scalls[__NR_getpmsg] = "getpmsg";
    __scalls[__NR_putpmsg] = "putpmsg";
    __scalls[__NR_vfork] = "vfork";
    __scalls[__NR_ugetrlimit] = "ugetrlimit";
    __scalls[__NR_readahead] = "readahead";
#ifndef __powerpc64__
    __scalls[__NR_mmap2] = "mmap2";
    __scalls[__NR_truncate64] = "truncate64";
    __scalls[__NR_ftruncate64] = "ftruncate64";
    __scalls[__NR_stat64] = "stat64";
    __scalls[__NR_lstat64] = "lstat64";
    __scalls[__NR_fstat64] = "fstat64";
#endif
    __scalls[__NR_pciconfig_read] = "pciconfig_read";
    __scalls[__NR_pciconfig_write] = "pciconfig_write";
    __scalls[__NR_pciconfig_iobase] = "pciconfig_iobase";
    __scalls[__NR_multiplexer] = "multiplexer";
    __scalls[__NR_getdents64] = "getdents64";
    __scalls[__NR_pivot_root] = "pivot_root";
#ifndef __powerpc64__
    __scalls[__NR_fcntl64] = "fcntl64";
#endif
    __scalls[__NR_madvise] = "madvise";
    __scalls[__NR_mincore] = "mincore";
    __scalls[__NR_gettid] = "gettid";
    __scalls[__NR_tkill] = "tkill";
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
    __scalls[__NR_futex] = "futex";
    __scalls[__NR_sched_setaffinity] = "sched_setaffinity";
    __scalls[__NR_sched_getaffinity] = "sched_getaffinity";

    __scalls[__NR_tuxcall] = "tuxcall";
#ifndef __powerpc64__
    __scalls[__NR_sendfile64] = "sendfile64";
#endif
    __scalls[__NR_io_setup] = "io_setup";
    __scalls[__NR_io_destroy] = "io_destroy";
    __scalls[__NR_io_getevents] = "io_getevents";
    __scalls[__NR_io_submit] = "io_submit";
    __scalls[__NR_io_cancel] = "io_cancel";
    __scalls[__NR_set_tid_address] = "set_tid_address";
    __scalls[__NR_fadvise64] = "fadvise64";
    __scalls[__NR_exit_group] = "exit_group";
    __scalls[__NR_lookup_dcookie] = "lookup_dcookie";
    __scalls[__NR_epoll_create] = "epoll_create";
    __scalls[__NR_epoll_ctl] = "epoll_ctl";
    __scalls[__NR_epoll_wait] = "epoll_wait";
    __scalls[__NR_remap_file_pages] = "remap_file_pages";
    __scalls[__NR_timer_create] = "timer_create";
    __scalls[__NR_timer_settime] = "timer_settime";
    __scalls[__NR_timer_gettime] = "timer_gettime";
    __scalls[__NR_timer_getoverrun] = "timer_getoverrun";
    __scalls[__NR_timer_delete] = "timer_delete";
    __scalls[__NR_clock_settime] = "clock_settime";
    __scalls[__NR_clock_gettime] = "clock_gettime";
    __scalls[__NR_clock_getres] = "clock_getres";
    __scalls[__NR_clock_nanosleep] = "clock_nanosleep";
    __scalls[__NR_swapcontext] = "swapcontext";
    __scalls[__NR_tgkill] = "tgkill";
    __scalls[__NR_utimes] = "utimes";
    __scalls[__NR_statfs64] = "statfs64";
    __scalls[__NR_fstatfs64] = "fstatfs64";
#ifndef __powerpc64__
    __scalls[__NR_fadvise64_64] = "fadvise64_64";
#endif
    __scalls[__NR_rtas] = "rtas";
    __scalls[__NR_sys_debug_setcontext] = "sys_debug_setcontext";

    __scalls[__NR_migrate_pages] = "migrate_pages";
    __scalls[__NR_mbind] = "mbind";
    __scalls[__NR_get_mempolicy] = "get_mempolicy";
    __scalls[__NR_set_mempolicy] = "set_mempolicy";
    __scalls[__NR_mq_open] = "mq_open";
    __scalls[__NR_mq_unlink] = "mq_unlink";
    __scalls[__NR_mq_timedsend] = "mq_timedsend";
    __scalls[__NR_mq_timedreceive] = "mq_timedreceive";
    __scalls[__NR_mq_notify] = "mq_notify";
    __scalls[__NR_mq_getsetattr] = "mq_getsetattr";
    __scalls[__NR_kexec_load] = "kexec_load";
    __scalls[__NR_add_key] = "add_key";
    __scalls[__NR_request_key] = "request_key";
    __scalls[__NR_keyctl] = "keyctl";
    __scalls[__NR_waitid] = "waitid";
    __scalls[__NR_ioprio_set] = "ioprio_set";
    __scalls[__NR_ioprio_get] = "ioprio_get";
    __scalls[__NR_inotify_init] = "inotify_init";
    __scalls[__NR_inotify_add_watch] = "inotify_add_watch";
    __scalls[__NR_inotify_rm_watch] = "inotify_rm_watch";
    __scalls[__NR_spu_run] = "spu_run";
    __scalls[__NR_spu_create] = "spu_create";
    __scalls[__NR_pselect6] = "pselect6";
    __scalls[__NR_ppoll] = "ppoll";
    __scalls[__NR_unshare] = "unshare";
    __scalls[__NR_splice] = "splice";
    __scalls[__NR_tee] = "tee";
    __scalls[__NR_vmsplice] = "vmsplice";
    __scalls[__NR_openat] = "openat";
    __scalls[__NR_mkdirat] = "mkdirat";
    __scalls[__NR_mknodat] = "mknodat";
    __scalls[__NR_fchownat] = "fchownat";
    __scalls[__NR_futimesat] = "futimesat";
#ifdef __powerpc64__
    __scalls[__NR_newfstatat] = "newfstatat";
#else
    __scalls[__NR_fstatat64] = "fstatat64";
#endif
    __scalls[__NR_unlinkat] = "unlinkat";
    __scalls[__NR_renameat] = "renameat";
    __scalls[__NR_linkat] = "linkat";
    __scalls[__NR_symlinkat] = "symlinkat";
    __scalls[__NR_readlinkat] = "readlinkat";
    __scalls[__NR_fchmodat] = "fchmodat";
    __scalls[__NR_faccessat] = "faccessat";
    __scalls[__NR_get_robust_list] = "get_robust_list";
    __scalls[__NR_set_robust_list] = "set_robust_list";
    __scalls[__NR_move_pages] = "move_pages";
    __scalls[__NR_getcpu] = "getcpu";
    __scalls[__NR_epoll_pwait] = "epoll_pwait";
    __scalls[__NR_utimensat] = "utimensat";
    __scalls[__NR_signalfd] = "signalfd";
    __scalls[__NR_timerfd_create] = "timerfd_create";
    __scalls[__NR_eventfd] = "eventfd";
    __scalls[__NR_sync_file_range2] = "sync_file_range2";
    __scalls[__NR_fallocate] = "fallocate";
    __scalls[__NR_subpage_prot] = "subpage_prot";
    __scalls[__NR_timerfd_settime] = "timerfd_settime";
    __scalls[__NR_timerfd_gettime] = "timerfd_gettime";
    __scalls[__NR_signalfd4] = "signalfd4";
    __scalls[__NR_eventfd2] = "eventfd2";
    __scalls[__NR_epoll_create1] = "epoll_create1";
    __scalls[__NR_dup3] = "dup3";
    __scalls[__NR_pipe2] = "pipe2";
    __scalls[__NR_inotify_init1] = "inotify_init1";
    __scalls[__NR_perf_event_open] = "perf_event_open";
    __scalls[__NR_preadv] = "preadv";
    __scalls[__NR_pwritev] = "pwritev";
    __scalls[__NR_rt_tgsigqueueinfo] = "rt_tgsigqueueinfo";
    __scalls[__NR_fanotify_init] = "fanotify_init";
    __scalls[__NR_fanotify_mark] = "fanotify_mark";
    __scalls[__NR_prlimit64] = "prlimit64";
    __scalls[__NR_socket] = "socket";
    __scalls[__NR_bind] = "bind";
    __scalls[__NR_connect] = "connect";
    __scalls[__NR_listen] = "listen";
    __scalls[__NR_accept] = "accept";
    __scalls[__NR_getsockname] = "getsockname";
    __scalls[__NR_getpeername] = "getpeername";
    __scalls[__NR_socketpair] = "socketpair";
    __scalls[__NR_send] = "send";
    __scalls[__NR_sendto] = "sendto";
    __scalls[__NR_recv] = "recv";
    __scalls[__NR_recvfrom] = "recvfrom";
    __scalls[__NR_shutdown] = "shutdown";
    __scalls[__NR_setsockopt] = "setsockopt";
    __scalls[__NR_getsockopt] = "getsockopt";
    __scalls[__NR_sendmsg] = "sendmsg";
    __scalls[__NR_recvmsg] = "recvmsg";
    __scalls[__NR_recvmmsg] = "recvmmsg";
    __scalls[__NR_accept4] = "accept4";
    __scalls[__NR_name_to_handle_at] = "name_to_handle_at";
    __scalls[__NR_open_by_handle_at] = "open_by_handle_at";
    __scalls[__NR_clock_adjtime] = "clock_adjtime";
    __scalls[__NR_syncfs] = "syncfs";
    __scalls[__NR_sendmmsg] = "sendmmsg";
    __scalls[__NR_setns] = "setns";
    __scalls[__NR_process_vm_readv] = "process_vm_readv";
    __scalls[__NR_process_vm_writev] = "process_vm_writev";
    __scalls[__NR_finit_module] = "finit_module";
    __scalls[__NR_kcmp] = "kcmp";
    __scalls[__NR_sched_setattr] = "sched_setattr";
    __scalls[__NR_sched_getattr] = "sched_getattr";
    __scalls[__NR_renameat2] = "renameat2";
    __scalls[__NR_seccomp] = "seccomp";
    __scalls[__NR_getrandom] = "getrandom";
    __scalls[__NR_memfd_create] = "memfd_create";
    __scalls[__NR_bpf] = "bpf";
    __scalls[__NR_execveat] = "execveat";
    __scalls[__NR_switch_endian] = "switch_endian";
    __scalls[__NR_userfaultfd] = "userfaultfd";
    __scalls[__NR_membarrier] = "membarrier";
    __scalls[__NR_mlock2] = "mlock2";
    __scalls[__NR_copy_file_range] = "copy_file_range";
    __scalls[__NR_preadv2] = "preadv2";
    __scalls[__NR_pwritev2] = "pwritev2";
    __scalls[__NR_kexec_file_load] = "kexec_file_load";
    __scalls[__NR_statx] = "statx";
    __scalls[__NR_pkey_alloc] = "pkey_alloc";
    __scalls[__NR_pkey_free] = "pkey_free";
    __scalls[__NR_pkey_mprotect] = "pkey_mprotect";
    __scalls[__NR_rseq] = "rseq";
    __scalls[__NR_io_pgetevents] = "io_pgetevents";
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
    __scalls[__NR_close_range] = "close_range";
    __scalls[__NR_openat2] = "openat2";
    __scalls[__NR_faccessat2] = "faccessat2";
}

static void printf_regs(struct pt_regs regs)
{
    for (int i = 0; i < 32; i++)
    {
        printf("gpr[%d] : %ld\n", i, regs.gpr[i]);
    }

    printf("nip : %ld\n", regs.nip);
    printf("msr : %ld\n", regs.msr);
    printf("orig_gpr3 : %ld\n", regs.orig_gpr3);
    printf("ctr : %ld\n", regs.ctr);
    printf("link : %ld\n", regs.link);
    printf("xer : %ld\n", regs.xer);
    printf("ccr : %ld\n", regs.ccr);

#ifdef __powerpc64__
    printf("softe : %ld\n", regs.softe);
#else
    printf("mq : %ld\n", regs.mq);
#endif
    printf("trap : %ld\n", regs.trap);
    printf("dar : %ld\n", regs.dar);
    printf("dsisr : %ld\n", regs.dsisr);
    printf("result : %ld\n", regs.result);
}

int get_syscall_id(pid_t pid)
{
    struct pt_regs regs;

    if (ptrace(PTRACE_GETREGS, pid, NULL, &regs) < 0)
    {
        perror("ptrace (getregs)");
        return -1;
    }

    return regs.gpr[0];
}

#endif
