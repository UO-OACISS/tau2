#ifdef __x86_64__

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
    switch(id)
    {
    #ifdef SYS__sysctl
      case SYS__sysctl : return "_sysctl";
    #endif

    #ifdef SYS_access
      case SYS_access : return "access";
    #endif

    #ifdef SYS_acct
      case SYS_acct : return "acct";
    #endif

    #ifdef SYS_add_key
      case SYS_add_key : return "add_key";
    #endif

    #ifdef SYS_adjtimex
      case SYS_adjtimex : return "adjtimex";
    #endif

    #ifdef SYS_afs_syscall
      case SYS_afs_syscall : return "afs_syscall";
    #endif

    #ifdef SYS_alarm
      case SYS_alarm : return "alarm";
    #endif

    #ifdef SYS_brk
      case SYS_brk : return "brk";
    #endif

    #ifdef SYS_capget
      case SYS_capget : return "capget";
    #endif

    #ifdef SYS_capset
      case SYS_capset : return "capset";
    #endif

    #ifdef SYS_chdir
      case SYS_chdir : return "chdir";
    #endif

    #ifdef SYS_chmod
      case SYS_chmod : return "chmod";
    #endif

    #ifdef SYS_chown
      case SYS_chown : return "chown";
    #endif

    #ifdef SYS_chroot
      case SYS_chroot : return "chroot";
    #endif

    #ifdef SYS_clock_getres
      case SYS_clock_getres : return "clock_getres";
    #endif

    #ifdef SYS_clock_gettime
      case SYS_clock_gettime : return "clock_gettime";
    #endif

    #ifdef SYS_clock_nanosleep
      case SYS_clock_nanosleep : return "clock_nanosleep";
    #endif

    #ifdef SYS_clock_settime
      case SYS_clock_settime : return "clock_settime";
    #endif

    #ifdef SYS_clone
      case SYS_clone : return "clone";
    #endif

    #ifdef SYS_close
      case SYS_close : return "close";
    #endif

    #ifdef SYS_creat
      case SYS_creat : return "creat";
    #endif

    #ifdef SYS_create_module
      case SYS_create_module : return "create_module";
    #endif

    #ifdef SYS_delete_module
      case SYS_delete_module : return "delete_module";
    #endif

    #ifdef SYS_dup
      case SYS_dup : return "dup";
    #endif

    #ifdef SYS_dup2
      case SYS_dup2 : return "dup2";
    #endif

    #ifdef SYS_epoll_create
      case SYS_epoll_create : return "epoll_create";
    #endif

    #ifdef SYS_epoll_ctl
      case SYS_epoll_ctl : return "epoll_ctl";
    #endif

    #ifdef SYS_epoll_pwait
      case SYS_epoll_pwait : return "epoll_pwait";
    #endif

    #ifdef SYS_epoll_wait
      case SYS_epoll_wait : return "epoll_wait";
    #endif

    #ifdef SYS_eventfd
      case SYS_eventfd : return "eventfd";
    #endif

    #ifdef SYS_execve
      case SYS_execve : return "execve";
    #endif

    #ifdef SYS_exit
      case SYS_exit : return "exit";
    #endif

    #ifdef SYS_exit_group
      case SYS_exit_group : return "exit_group";
    #endif

    #ifdef SYS_faccessat
      case SYS_faccessat : return "faccessat";
    #endif

    #ifdef SYS_fadvise64
      case SYS_fadvise64 : return "fadvise64";
    #endif

    #ifdef SYS_fallocate
      case SYS_fallocate : return "fallocate";
    #endif

    #ifdef SYS_fchdir
      case SYS_fchdir : return "fchdir";
    #endif

    #ifdef SYS_fchmod
      case SYS_fchmod : return "fchmod";
    #endif

    #ifdef SYS_fchmodat
      case SYS_fchmodat : return "fchmodat";
    #endif

    #ifdef SYS_fchown
      case SYS_fchown : return "fchown";
    #endif

    #ifdef SYS_fchownat
      case SYS_fchownat : return "fchownat";
    #endif

    #ifdef SYS_fcntl
      case SYS_fcntl : return "fcntl";
    #endif

    #ifdef SYS_fdatasync
      case SYS_fdatasync : return "fdatasync";
    #endif

    #ifdef SYS_fgetxattr
      case SYS_fgetxattr : return "fgetxattr";
    #endif

    #ifdef SYS_flistxattr
      case SYS_flistxattr : return "flistxattr";
    #endif

    #ifdef SYS_flock
      case SYS_flock : return "flock";
    #endif

    #ifdef SYS_fork
      case SYS_fork : return "fork";
    #endif

    #ifdef SYS_fremovexattr
      case SYS_fremovexattr : return "fremovexattr";
    #endif

    #ifdef SYS_fsetxattr
      case SYS_fsetxattr : return "fsetxattr";
    #endif

    #ifdef SYS_fstat
      case SYS_fstat : return "fstat";
    #endif

    #ifdef SYS_fstatfs
      case SYS_fstatfs : return "fstatfs";
    #endif

    #ifdef SYS_fsync
      case SYS_fsync : return "fsync";
    #endif

    #ifdef SYS_ftruncate
      case SYS_ftruncate : return "ftruncate";
    #endif

    #ifdef SYS_futex
      case SYS_futex : return "futex";
    #endif

    #ifdef SYS_futimesat
      case SYS_futimesat : return "futimesat";
    #endif

    #ifdef SYS_get_kernel_syms
      case SYS_get_kernel_syms : return "get_kernel_syms";
    #endif

    #ifdef SYS_get_mempolicy
      case SYS_get_mempolicy : return "get_mempolicy";
    #endif

    #ifdef SYS_get_robust_list
      case SYS_get_robust_list : return "get_robust_list";
    #endif

    #ifdef SYS_get_thread_area
      case SYS_get_thread_area : return "get_thread_area";
    #endif

    #ifdef SYS_getcwd
      case SYS_getcwd : return "getcwd";
    #endif

    #ifdef SYS_getdents
      case SYS_getdents : return "getdents";
    #endif

    #ifdef SYS_getdents64
      case SYS_getdents64 : return "getdents64";
    #endif

    #ifdef SYS_getegid
      case SYS_getegid : return "getegid";
    #endif

    #ifdef SYS_geteuid
      case SYS_geteuid : return "geteuid";
    #endif

    #ifdef SYS_getgid
      case SYS_getgid : return "getgid";
    #endif

    #ifdef SYS_getgroups
      case SYS_getgroups : return "getgroups";
    #endif

    #ifdef SYS_getitimer
      case SYS_getitimer : return "getitimer";
    #endif

    #ifdef SYS_getpgid
      case SYS_getpgid : return "getpgid";
    #endif

    #ifdef SYS_getpgrp
      case SYS_getpgrp : return "getpgrp";
    #endif

    #ifdef SYS_getpid
      case SYS_getpid : return "getpid";
    #endif

    #ifdef SYS_getpmsg
      case SYS_getpmsg : return "getpmsg";
    #endif

    #ifdef SYS_getppid
      case SYS_getppid : return "getppid";
    #endif

    #ifdef SYS_getpriority
      case SYS_getpriority : return "getpriority";
    #endif

    #ifdef SYS_getresgid
      case SYS_getresgid : return "getresgid";
    #endif

    #ifdef SYS_getresuid
      case SYS_getresuid : return "getresuid";
    #endif

    #ifdef SYS_getrlimit
      case SYS_getrlimit : return "getrlimit";
    #endif

    #ifdef SYS_getrusage
      case SYS_getrusage : return "getrusage";
    #endif

    #ifdef SYS_getsid
      case SYS_getsid : return "getsid";
    #endif

    #ifdef SYS_gettid
      case SYS_gettid : return "gettid";
    #endif

    #ifdef SYS_gettimeofday
      case SYS_gettimeofday : return "gettimeofday";
    #endif

    #ifdef SYS_getuid
      case SYS_getuid : return "getuid";
    #endif

    #ifdef SYS_getxattr
      case SYS_getxattr : return "getxattr";
    #endif

    #ifdef SYS_init_module
      case SYS_init_module : return "init_module";
    #endif

    #ifdef SYS_inotify_add_watch
      case SYS_inotify_add_watch : return "inotify_add_watch";
    #endif

    #ifdef SYS_inotify_init
      case SYS_inotify_init : return "inotify_init";
    #endif

    #ifdef SYS_inotify_rm_watch
      case SYS_inotify_rm_watch : return "inotify_rm_watch";
    #endif

    #ifdef SYS_io_cancel
      case SYS_io_cancel : return "io_cancel";
    #endif

    #ifdef SYS_io_destroy
      case SYS_io_destroy : return "io_destroy";
    #endif

    #ifdef SYS_io_getevents
      case SYS_io_getevents : return "io_getevents";
    #endif

    #ifdef SYS_io_setup
      case SYS_io_setup : return "io_setup";
    #endif

    #ifdef SYS_io_submit
      case SYS_io_submit : return "io_submit";
    #endif

    #ifdef SYS_ioctl
      case SYS_ioctl : return "ioctl";
    #endif

    #ifdef SYS_ioperm
      case SYS_ioperm : return "ioperm";
    #endif

    #ifdef SYS_iopl
      case SYS_iopl : return "iopl";
    #endif

    #ifdef SYS_ioprio_get
      case SYS_ioprio_get : return "ioprio_get";
    #endif

    #ifdef SYS_ioprio_set
      case SYS_ioprio_set : return "ioprio_set";
    #endif

    #ifdef SYS_kexec_load
      case SYS_kexec_load : return "kexec_load";
    #endif

    #ifdef SYS_keyctl
      case SYS_keyctl : return "keyctl";
    #endif

    #ifdef SYS_kill
      case SYS_kill : return "kill";
    #endif

    #ifdef SYS_lchown
      case SYS_lchown : return "lchown";
    #endif

    #ifdef SYS_lgetxattr
      case SYS_lgetxattr : return "lgetxattr";
    #endif

    #ifdef SYS_link
      case SYS_link : return "link";
    #endif

    #ifdef SYS_linkat
      case SYS_linkat : return "linkat";
    #endif

    #ifdef SYS_listxattr
      case SYS_listxattr : return "listxattr";
    #endif

    #ifdef SYS_llistxattr
      case SYS_llistxattr : return "llistxattr";
    #endif

    #ifdef SYS_lookup_dcookie
      case SYS_lookup_dcookie : return "lookup_dcookie";
    #endif

    #ifdef SYS_lremovexattr
      case SYS_lremovexattr : return "lremovexattr";
    #endif

    #ifdef SYS_lseek
      case SYS_lseek : return "lseek";
    #endif

    #ifdef SYS_lsetxattr
      case SYS_lsetxattr : return "lsetxattr";
    #endif

    #ifdef SYS_lstat
      case SYS_lstat : return "lstat";
    #endif

    #ifdef SYS_madvise
      case SYS_madvise : return "madvise";
    #endif

    #ifdef SYS_mbind
      case SYS_mbind : return "mbind";
    #endif

    #ifdef SYS_migrate_pages
      case SYS_migrate_pages : return "migrate_pages";
    #endif

    #ifdef SYS_mincore
      case SYS_mincore : return "mincore";
    #endif

    #ifdef SYS_mkdir
      case SYS_mkdir : return "mkdir";
    #endif

    #ifdef SYS_mkdirat
      case SYS_mkdirat : return "mkdirat";
    #endif

    #ifdef SYS_mknod
      case SYS_mknod : return "mknod";
    #endif

    #ifdef SYS_mknodat
      case SYS_mknodat : return "mknodat";
    #endif

    #ifdef SYS_mlock
      case SYS_mlock : return "mlock";
    #endif

    #ifdef SYS_mlockall
      case SYS_mlockall : return "mlockall";
    #endif

    #ifdef SYS_mmap
      case SYS_mmap : return "mmap";
    #endif

    #ifdef SYS_modify_ldt
      case SYS_modify_ldt : return "modify_ldt";
    #endif

    #ifdef SYS_mount
      case SYS_mount : return "mount";
    #endif

    #ifdef SYS_move_pages
      case SYS_move_pages : return "move_pages";
    #endif

    #ifdef SYS_mprotect
      case SYS_mprotect : return "mprotect";
    #endif

    #ifdef SYS_mq_getsetattr
      case SYS_mq_getsetattr : return "mq_getsetattr";
    #endif

    #ifdef SYS_mq_notify
      case SYS_mq_notify : return "mq_notify";
    #endif

    #ifdef SYS_mq_open
      case SYS_mq_open : return "mq_open";
    #endif

    #ifdef SYS_mq_timedreceive
      case SYS_mq_timedreceive : return "mq_timedreceive";
    #endif

    #ifdef SYS_mq_timedsend
      case SYS_mq_timedsend : return "mq_timedsend";
    #endif

    #ifdef SYS_mq_unlink
      case SYS_mq_unlink : return "mq_unlink";
    #endif

    #ifdef SYS_mremap
      case SYS_mremap : return "mremap";
    #endif

    #ifdef SYS_msync
      case SYS_msync : return "msync";
    #endif

    #ifdef SYS_munlock
      case SYS_munlock : return "munlock";
    #endif

    #ifdef SYS_munlockall
      case SYS_munlockall : return "munlockall";
    #endif

    #ifdef SYS_munmap
      case SYS_munmap : return "munmap";
    #endif

    #ifdef SYS_nanosleep
      case SYS_nanosleep : return "nanosleep";
    #endif

    #ifdef SYS_nfsservctl
      case SYS_nfsservctl : return "nfsservctl";
    #endif

    #ifdef SYS_open
      case SYS_open : return "open";
    #endif

    #ifdef SYS_openat
      case SYS_openat : return "openat";
    #endif

    #ifdef SYS_pause
      case SYS_pause : return "pause";
    #endif

    #ifdef SYS_personality
      case SYS_personality : return "personality";
    #endif

    #ifdef SYS_pipe
      case SYS_pipe : return "pipe";
    #endif

    #ifdef SYS_pivot_root
      case SYS_pivot_root : return "pivot_root";
    #endif

    #ifdef SYS_poll
      case SYS_poll : return "poll";
    #endif

    #ifdef SYS_ppoll
      case SYS_ppoll : return "ppoll";
    #endif

    #ifdef SYS_prctl
      case SYS_prctl : return "prctl";
    #endif

    #ifdef SYS_pread64
      case SYS_pread64 : return "pread64";
    #endif

    #ifdef SYS_pselect6
      case SYS_pselect6 : return "pselect6";
    #endif

    #ifdef SYS_ptrace
      case SYS_ptrace : return "ptrace";
    #endif

    #ifdef SYS_putpmsg
      case SYS_putpmsg : return "putpmsg";
    #endif

    #ifdef SYS_pwrite64
      case SYS_pwrite64 : return "pwrite64";
    #endif

    #ifdef SYS_query_module
      case SYS_query_module : return "query_module";
    #endif

    #ifdef SYS_quotactl
      case SYS_quotactl : return "quotactl";
    #endif

    #ifdef SYS_read
      case SYS_read : return "read";
    #endif

    #ifdef SYS_readahead
      case SYS_readahead : return "readahead";
    #endif

    #ifdef SYS_readlink
      case SYS_readlink : return "readlink";
    #endif

    #ifdef SYS_readlinkat
      case SYS_readlinkat : return "readlinkat";
    #endif

    #ifdef SYS_readv
      case SYS_readv : return "readv";
    #endif

    #ifdef SYS_reboot
      case SYS_reboot : return "reboot";
    #endif

    #ifdef SYS_remap_file_pages
      case SYS_remap_file_pages : return "remap_file_pages";
    #endif

    #ifdef SYS_removexattr
      case SYS_removexattr : return "removexattr";
    #endif

    #ifdef SYS_rename
      case SYS_rename : return "rename";
    #endif

    #ifdef SYS_renameat
      case SYS_renameat : return "renameat";
    #endif

    #ifdef SYS_request_key
      case SYS_request_key : return "request_key";
    #endif

    #ifdef SYS_restart_syscall
      case SYS_restart_syscall : return "restart_syscall";
    #endif

    #ifdef SYS_rmdir
      case SYS_rmdir : return "rmdir";
    #endif

    #ifdef SYS_rt_sigaction
      case SYS_rt_sigaction : return "rt_sigaction";
    #endif

    #ifdef SYS_rt_sigpending
      case SYS_rt_sigpending : return "rt_sigpending";
    #endif

    #ifdef SYS_rt_sigprocmask
      case SYS_rt_sigprocmask : return "rt_sigprocmask";
    #endif

    #ifdef SYS_rt_sigqueueinfo
      case SYS_rt_sigqueueinfo : return "rt_sigqueueinfo";
    #endif

    #ifdef SYS_rt_sigreturn
      case SYS_rt_sigreturn : return "rt_sigreturn";
    #endif

    #ifdef SYS_rt_sigsuspend
      case SYS_rt_sigsuspend : return "rt_sigsuspend";
    #endif

    #ifdef SYS_rt_sigtimedwait
      case SYS_rt_sigtimedwait : return "rt_sigtimedwait";
    #endif

    #ifdef SYS_sched_get_priority_max
      case SYS_sched_get_priority_max : return "sched_get_priority_max";
    #endif

    #ifdef SYS_sched_get_priority_min
      case SYS_sched_get_priority_min : return "sched_get_priority_min";
    #endif

    #ifdef SYS_sched_getaffinity
      case SYS_sched_getaffinity : return "sched_getaffinity";
    #endif

    #ifdef SYS_sched_getparam
      case SYS_sched_getparam : return "sched_getparam";
    #endif

    #ifdef SYS_sched_getscheduler
      case SYS_sched_getscheduler : return "sched_getscheduler";
    #endif

    #ifdef SYS_sched_rr_get_interval
      case SYS_sched_rr_get_interval : return "sched_rr_get_interval";
    #endif

    #ifdef SYS_sched_setaffinity
      case SYS_sched_setaffinity : return "sched_setaffinity";
    #endif

    #ifdef SYS_sched_setparam
      case SYS_sched_setparam : return "sched_setparam";
    #endif

    #ifdef SYS_sched_setscheduler
      case SYS_sched_setscheduler : return "sched_setscheduler";
    #endif

    #ifdef SYS_sched_yield
      case SYS_sched_yield : return "sched_yield";
    #endif

    #ifdef SYS_select
      case SYS_select : return "select";
    #endif

    #ifdef SYS_sendfile
      case SYS_sendfile : return "sendfile";
    #endif

    #ifdef SYS_set_mempolicy
      case SYS_set_mempolicy : return "set_mempolicy";
    #endif

    #ifdef SYS_set_robust_list
      case SYS_set_robust_list : return "set_robust_list";
    #endif

    #ifdef SYS_set_thread_area
      case SYS_set_thread_area : return "set_thread_area";
    #endif

    #ifdef SYS_set_tid_address
      case SYS_set_tid_address : return "set_tid_address";
    #endif

    #ifdef SYS_setdomainname
      case SYS_setdomainname : return "setdomainname";
    #endif

    #ifdef SYS_setfsgid
      case SYS_setfsgid : return "setfsgid";
    #endif

    #ifdef SYS_setfsuid
      case SYS_setfsuid : return "setfsuid";
    #endif

    #ifdef SYS_setgid
      case SYS_setgid : return "setgid";
    #endif

    #ifdef SYS_setgroups
      case SYS_setgroups : return "setgroups";
    #endif

    #ifdef SYS_sethostname
      case SYS_sethostname : return "sethostname";
    #endif

    #ifdef SYS_setitimer
      case SYS_setitimer : return "setitimer";
    #endif

    #ifdef SYS_setpgid
      case SYS_setpgid : return "setpgid";
    #endif

    #ifdef SYS_setpriority
      case SYS_setpriority : return "setpriority";
    #endif

    #ifdef SYS_setregid
      case SYS_setregid : return "setregid";
    #endif

    #ifdef SYS_setresgid
      case SYS_setresgid : return "setresgid";
    #endif

    #ifdef SYS_setresuid
      case SYS_setresuid : return "setresuid";
    #endif

    #ifdef SYS_setreuid
      case SYS_setreuid : return "setreuid";
    #endif

    #ifdef SYS_setrlimit
      case SYS_setrlimit : return "setrlimit";
    #endif

    #ifdef SYS_setsid
      case SYS_setsid : return "setsid";
    #endif

    #ifdef SYS_settimeofday
      case SYS_settimeofday : return "settimeofday";
    #endif

    #ifdef SYS_setuid
      case SYS_setuid : return "setuid";
    #endif

    #ifdef SYS_setxattr
      case SYS_setxattr : return "setxattr";
    #endif

    #ifdef SYS_sigaltstack
      case SYS_sigaltstack : return "sigaltstack";
    #endif

    #ifdef SYS_signalfd
      case SYS_signalfd : return "signalfd";
    #endif

    #ifdef SYS_splice
      case SYS_splice : return "splice";
    #endif

    #ifdef SYS_stat
      case SYS_stat : return "stat";
    #endif

    #ifdef SYS_statfs
      case SYS_statfs : return "statfs";
    #endif

    #ifdef SYS_swapoff
      case SYS_swapoff : return "swapoff";
    #endif

    #ifdef SYS_swapon
      case SYS_swapon : return "swapon";
    #endif

    #ifdef SYS_symlink
      case SYS_symlink : return "symlink";
    #endif

    #ifdef SYS_symlinkat
      case SYS_symlinkat : return "symlinkat";
    #endif

    #ifdef SYS_sync
      case SYS_sync : return "sync";
    #endif

    #ifdef SYS_sync_file_range
      case SYS_sync_file_range : return "sync_file_range";
    #endif

    #ifdef SYS_sysfs
      case SYS_sysfs : return "sysfs";
    #endif

    #ifdef SYS_sysinfo
      case SYS_sysinfo : return "sysinfo";
    #endif

    #ifdef SYS_syslog
      case SYS_syslog : return "syslog";
    #endif

    #ifdef SYS_tee
      case SYS_tee : return "tee";
    #endif

    #ifdef SYS_tgkill
      case SYS_tgkill : return "tgkill";
    #endif

    #ifdef SYS_time
      case SYS_time : return "time";
    #endif

    #ifdef SYS_timer_create
      case SYS_timer_create : return "timer_create";
    #endif

    #ifdef SYS_timer_delete
      case SYS_timer_delete : return "timer_delete";
    #endif

    #ifdef SYS_timer_getoverrun
      case SYS_timer_getoverrun : return "timer_getoverrun";
    #endif

    #ifdef SYS_timer_gettime
      case SYS_timer_gettime : return "timer_gettime";
    #endif

    #ifdef SYS_timer_settime
      case SYS_timer_settime : return "timer_settime";
    #endif

    #ifdef SYS_timerfd_create
      case SYS_timerfd_create : return "timerfd_create";
    #endif

    #ifdef SYS_timerfd_gettime
      case SYS_timerfd_gettime : return "timerfd_gettime";
    #endif

    #ifdef SYS_timerfd_settime
      case SYS_timerfd_settime : return "timerfd_settime";
    #endif

    #ifdef SYS_times
      case SYS_times : return "times";
    #endif

    #ifdef SYS_tkill
      case SYS_tkill : return "tkill";
    #endif

    #ifdef SYS_truncate
      case SYS_truncate : return "truncate";
    #endif

    #ifdef SYS_umask
      case SYS_umask : return "umask";
    #endif

    #ifdef SYS_umount2
      case SYS_umount2 : return "umount2";
    #endif

    #ifdef SYS_uname
      case SYS_uname : return "uname";
    #endif

    #ifdef SYS_unlink
      case SYS_unlink : return "unlink";
    #endif

    #ifdef SYS_unlinkat
      case SYS_unlinkat : return "unlinkat";
    #endif

    #ifdef SYS_unshare
      case SYS_unshare : return "unshare";
    #endif

    #ifdef SYS_uselib
      case SYS_uselib : return "uselib";
    #endif

    #ifdef SYS_ustat
      case SYS_ustat : return "ustat";
    #endif

    #ifdef SYS_utime
      case SYS_utime : return "utime";
    #endif

    #ifdef SYS_utimensat
      case SYS_utimensat : return "utimensat";
    #endif

    #ifdef SYS_utimes
      case SYS_utimes : return "utimes";
    #endif

    #ifdef SYS_vfork
      case SYS_vfork : return "vfork";
    #endif

    #ifdef SYS_vhangup
      case SYS_vhangup : return "vhangup";
    #endif

    #ifdef SYS_vmsplice
      case SYS_vmsplice : return "vmsplice";
    #endif

    #ifdef SYS_vserver
      case SYS_vserver : return "vserver";
    #endif

    #ifdef SYS_wait4
      case SYS_wait4 : return "wait4";
    #endif

    #ifdef SYS_waitid
      case SYS_waitid : return "waitid";
    #endif

    #ifdef SYS_write
      case SYS_write : return "write";
    #endif

    #ifdef SYS_writev
      case SYS_writev : return "writev";
    #endif

    #ifdef SYS_accept
      case SYS_accept : return "accept";
    #endif

    #ifdef SYS_arch_prctl
      case SYS_arch_prctl : return "arch_prctl";
    #endif

    #ifdef SYS_bind
      case SYS_bind : return "bind";
    #endif

    #ifdef SYS_connect
      case SYS_connect : return "connect";
    #endif

    #ifdef SYS_epoll_ctl_old
      case SYS_epoll_ctl_old : return "epoll_ctl_old";
    #endif

    #ifdef SYS_epoll_wait_old
      case SYS_epoll_wait_old : return "epoll_wait_old";
    #endif

    #ifdef SYS_getpeername
      case SYS_getpeername : return "getpeername";
    #endif

    #ifdef SYS_getsockname
      case SYS_getsockname : return "getsockname";
    #endif

    #ifdef SYS_getsockopt
      case SYS_getsockopt : return "getsockopt";
    #endif

    #ifdef SYS_listen
      case SYS_listen : return "listen";
    #endif

    #ifdef SYS_msgctl
      case SYS_msgctl : return "msgctl";
    #endif

    #ifdef SYS_msgget
      case SYS_msgget : return "msgget";
    #endif

    #ifdef SYS_msgrcv
      case SYS_msgrcv : return "msgrcv";
    #endif

    #ifdef SYS_msgsnd
      case SYS_msgsnd : return "msgsnd";
    #endif

    #ifdef SYS_newfstatat
      case SYS_newfstatat : return "newfstatat";
    #endif

    #ifdef SYS_recvfrom
      case SYS_recvfrom : return "recvfrom";
    #endif

    #ifdef SYS_recvmsg
      case SYS_recvmsg : return "recvmsg";
    #endif

    #ifdef SYS_security
      case SYS_security : return "security";
    #endif

    #ifdef SYS_semctl
      case SYS_semctl : return "semctl";
    #endif

    #ifdef SYS_semget
      case SYS_semget : return "semget";
    #endif

    #ifdef SYS_semop
      case SYS_semop : return "semop";
    #endif

    #ifdef SYS_semtimedop
      case SYS_semtimedop : return "semtimedop";
    #endif

    #ifdef SYS_sendmsg
      case SYS_sendmsg : return "sendmsg";
    #endif

    #ifdef SYS_sendto
      case SYS_sendto : return "sendto";
    #endif

    #ifdef SYS_setsockopt
      case SYS_setsockopt : return "setsockopt";
    #endif

    #ifdef SYS_shmat
      case SYS_shmat : return "shmat";
    #endif

    #ifdef SYS_shmctl
      case SYS_shmctl : return "shmctl";
    #endif

    #ifdef SYS_shmdt
      case SYS_shmdt : return "shmdt";
    #endif

    #ifdef SYS_shmget
      case SYS_shmget : return "shmget";
    #endif

    #ifdef SYS_shutdown
      case SYS_shutdown : return "shutdown";
    #endif

    #ifdef SYS_socket
      case SYS_socket : return "socket";
    #endif

    #ifdef SYS_socketpair
      case SYS_socketpair : return "socketpair";
    #endif

    #ifdef SYS_tuxcall
      case SYS_tuxcall : return "tuxcall";
    #endif

    #ifdef SYS__llseek
      case SYS__llseek : return "_llseek";
    #endif

    #ifdef SYS__newselect
      case SYS__newselect : return "_newselect";
    #endif

    #ifdef SYS_bdflush
      case SYS_bdflush : return "bdflush";
    #endif

    #ifdef SYS_break
      case SYS_break : return "break";
    #endif

    #ifdef SYS_chown32
      case SYS_chown32 : return "chown32";
    #endif

    #ifdef SYS_fadvise64_64
      case SYS_fadvise64_64 : return "fadvise64_64";
    #endif

    #ifdef SYS_fchown32
      case SYS_fchown32 : return "fchown32";
    #endif

    #ifdef SYS_fcntl64
      case SYS_fcntl64 : return "fcntl64";
    #endif

    #ifdef SYS_fstat64
      case SYS_fstat64 : return "fstat64";
    #endif

    #ifdef SYS_fstatat64
      case SYS_fstatat64 : return "fstatat64";
    #endif

    #ifdef SYS_fstatfs64
      case SYS_fstatfs64 : return "fstatfs64";
    #endif

    #ifdef SYS_ftime
      case SYS_ftime : return "ftime";
    #endif

    #ifdef SYS_ftruncate64
      case SYS_ftruncate64 : return "ftruncate64";
    #endif

    #ifdef SYS_getcpu
      case SYS_getcpu : return "getcpu";
    #endif

    #ifdef SYS_getegid32
      case SYS_getegid32 : return "getegid32";
    #endif

    #ifdef SYS_geteuid32
      case SYS_geteuid32 : return "geteuid32";
    #endif

    #ifdef SYS_getgid32
      case SYS_getgid32 : return "getgid32";
    #endif

    #ifdef SYS_getgroups32
      case SYS_getgroups32 : return "getgroups32";
    #endif

    #ifdef SYS_getresgid32
      case SYS_getresgid32 : return "getresgid32";
    #endif

    #ifdef SYS_getresuid32
      case SYS_getresuid32 : return "getresuid32";
    #endif

    #ifdef SYS_getuid32
      case SYS_getuid32 : return "getuid32";
    #endif

    #ifdef SYS_gtty
      case SYS_gtty : return "gtty";
    #endif

    #ifdef SYS_idle
      case SYS_idle : return "idle";
    #endif

    #ifdef SYS_ipc
      case SYS_ipc : return "ipc";
    #endif

    #ifdef SYS_lchown32
      case SYS_lchown32 : return "lchown32";
    #endif

    #ifdef SYS_lock
      case SYS_lock : return "lock";
    #endif

    #ifdef SYS_lstat64
      case SYS_lstat64 : return "lstat64";
    #endif

    #ifdef SYS_madvise1
      case SYS_madvise1 : return "madvise1";
    #endif

    #ifdef SYS_mmap2
      case SYS_mmap2 : return "mmap2";
    #endif

    #ifdef SYS_mpx
      case SYS_mpx : return "mpx";
    #endif

    #ifdef SYS_nice
      case SYS_nice : return "nice";
    #endif

    #ifdef SYS_oldfstat
      case SYS_oldfstat : return "oldfstat";
    #endif

    #ifdef SYS_oldlstat
      case SYS_oldlstat : return "oldlstat";
    #endif

    #ifdef SYS_oldolduname
      case SYS_oldolduname : return "oldolduname";
    #endif

    #ifdef SYS_oldstat
      case SYS_oldstat : return "oldstat";
    #endif

    #ifdef SYS_olduname
      case SYS_olduname : return "olduname";
    #endif

    #ifdef SYS_prof
      case SYS_prof : return "prof";
    #endif

    #ifdef SYS_profil
      case SYS_profil : return "profil";
    #endif

    #ifdef SYS_readdir
      case SYS_readdir : return "readdir";
    #endif

    #ifdef SYS_sendfile64
      case SYS_sendfile64 : return "sendfile64";
    #endif

    #ifdef SYS_setfsgid32
      case SYS_setfsgid32 : return "setfsgid32";
    #endif

    #ifdef SYS_setfsuid32
      case SYS_setfsuid32 : return "setfsuid32";
    #endif

    #ifdef SYS_setgid32
      case SYS_setgid32 : return "setgid32";
    #endif

    #ifdef SYS_setgroups32
      case SYS_setgroups32 : return "setgroups32";
    #endif

    #ifdef SYS_setregid32
      case SYS_setregid32 : return "setregid32";
    #endif

    #ifdef SYS_setresgid32
      case SYS_setresgid32 : return "setresgid32";
    #endif

    #ifdef SYS_setresuid32
      case SYS_setresuid32 : return "setresuid32";
    #endif

    #ifdef SYS_setreuid32
      case SYS_setreuid32 : return "setreuid32";
    #endif

    #ifdef SYS_setuid32
      case SYS_setuid32 : return "setuid32";
    #endif

    #ifdef SYS_sgetmask
      case SYS_sgetmask : return "sgetmask";
    #endif

    #ifdef SYS_sigaction
      case SYS_sigaction : return "sigaction";
    #endif

    #ifdef SYS_signal
      case SYS_signal : return "signal";
    #endif

    #ifdef SYS_sigpending
      case SYS_sigpending : return "sigpending";
    #endif

    #ifdef SYS_sigprocmask
      case SYS_sigprocmask : return "sigprocmask";
    #endif

    #ifdef SYS_sigreturn
      case SYS_sigreturn : return "sigreturn";
    #endif

    #ifdef SYS_sigsuspend
      case SYS_sigsuspend : return "sigsuspend";
    #endif

    #ifdef SYS_socketcall
      case SYS_socketcall : return "socketcall";
    #endif

    #ifdef SYS_ssetmask
      case SYS_ssetmask : return "ssetmask";
    #endif

    #ifdef SYS_stat64
      case SYS_stat64 : return "stat64";
    #endif

    #ifdef SYS_statfs64
      case SYS_statfs64 : return "statfs64";
    #endif

    #ifdef SYS_stime
      case SYS_stime : return "stime";
    #endif

    #ifdef SYS_stty
      case SYS_stty : return "stty";
    #endif

    #ifdef SYS_truncate64
      case SYS_truncate64 : return "truncate64";
    #endif

    #ifdef SYS_ugetrlimit
      case SYS_ugetrlimit : return "ugetrlimit";
    #endif

    #ifdef SYS_ulimit
      case SYS_ulimit : return "ulimit";
    #endif

    #ifdef SYS_umount
      case SYS_umount : return "umount";
    #endif

    #ifdef SYS_vm86
      case SYS_vm86 : return "vm86";
    #endif

    #ifdef SYS_vm86old
      case SYS_vm86old : return "vm86old";
    #endif

    #ifdef SYS_waitpid
      case SYS_waitpid : return "waitpid";
    #endif

    default:
        return "unknown";
  }
}

/**********************
 * ARGUMENTS HANDLING *
 **********************/

int get_syscall_id(pid_t pid)
{
    struct user_regs_struct regs;

    if (ptrace(PTRACE_GETREGS, pid, NULL, &regs) < 0)
    {
        return -1;
    }

    return regs.orig_rax;
}

#endif
