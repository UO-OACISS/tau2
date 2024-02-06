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
    switch(id)
    {
        #ifdef  __NR_exit
         case __NR_exit : return "exit";
        #endif
        #ifdef  __NR_restart_syscall
         case __NR_restart_syscall : return "restart_syscall";
        #endif
        #ifdef  __NR_exit
         case __NR_exit : return "exit";
        #endif
        #ifdef  __NR_fork
         case __NR_fork : return "fork";
        #endif
        #ifdef  __NR_read
         case __NR_read : return "read";
        #endif
        #ifdef  __NR_write
         case __NR_write : return "write";
        #endif
        #ifdef  __NR_open
         case __NR_open : return "open";
        #endif
        #ifdef  __NR_close
         case __NR_close : return "close";
        #endif
        #ifdef  __NR_waitpid
         case __NR_waitpid : return "waitpid";
        #endif
        #ifdef  __NR_creat
         case __NR_creat : return "creat";
        #endif
        #ifdef  __NR_link
         case __NR_link : return "link";
        #endif
        #ifdef  __NR_unlink
         case __NR_unlink : return "unlink";
        #endif
        #ifdef  __NR_execve
         case __NR_execve : return "execve";
        #endif
        #ifdef  __NR_chdir
         case __NR_chdir : return "chdir";
        #endif
        #ifdef  __NR_time
         case __NR_time : return "time";
        #endif
        #ifdef  __NR_mknod
         case __NR_mknod : return "mknod";
        #endif
        #ifdef  __NR_chmod
         case __NR_chmod : return "chmod";
        #endif
        #ifdef  __NR_lchown
         case __NR_lchown : return "lchown";
        #endif
        #ifdef  __NR_break
         case __NR_break : return "break";
        #endif
        #ifdef  __NR_oldstat
         case __NR_oldstat : return "oldstat";
        #endif
        #ifdef  __NR_lseek
         case __NR_lseek : return "lseek";
        #endif
        #ifdef  __NR_getpid
         case __NR_getpid : return "getpid";
        #endif
        #ifdef  __NR_mount
         case __NR_mount : return "mount";
        #endif
        #ifdef  __NR_umount
         case __NR_umount : return "umount";
        #endif
        #ifdef  __NR_setuid
         case __NR_setuid : return "setuid";
        #endif
        #ifdef  __NR_getuid
         case __NR_getuid : return "getuid";
        #endif
        #ifdef  __NR_stime
         case __NR_stime : return "stime";
        #endif
        #ifdef  __NR_ptrace
         case __NR_ptrace : return "ptrace";
        #endif
        #ifdef  __NR_alarm
         case __NR_alarm : return "alarm";
        #endif
        #ifdef  __NR_oldfstat
         case __NR_oldfstat : return "oldfstat";
        #endif
        #ifdef  __NR_pause
         case __NR_pause : return "pause";
        #endif
        #ifdef  __NR_utime
         case __NR_utime : return "utime";
        #endif
        #ifdef  __NR_stty
         case __NR_stty : return "stty";
        #endif
        #ifdef  __NR_gtty
         case __NR_gtty : return "gtty";
        #endif
        #ifdef  __NR_access
         case __NR_access : return "access";
        #endif
        #ifdef  __NR_nice
         case __NR_nice : return "nice";
        #endif
        #ifdef  __NR_ftime
         case __NR_ftime : return "ftime";
        #endif
        #ifdef  __NR_sync
         case __NR_sync : return "sync";
        #endif
        #ifdef  __NR_kill
         case __NR_kill : return "kill";
        #endif
        #ifdef  __NR_rename
         case __NR_rename : return "rename";
        #endif
        #ifdef  __NR_mkdir
         case __NR_mkdir : return "mkdir";
        #endif
        #ifdef  __NR_rmdir
         case __NR_rmdir : return "rmdir";
        #endif
        #ifdef  __NR_dup
         case __NR_dup : return "dup";
        #endif
        #ifdef  __NR_pipe
         case __NR_pipe : return "pipe";
        #endif
        #ifdef  __NR_times
         case __NR_times : return "times";
        #endif
        #ifdef  __NR_prof
         case __NR_prof : return "prof";
        #endif
        #ifdef  __NR_brk
         case __NR_brk : return "brk";
        #endif
        #ifdef  __NR_setgid
         case __NR_setgid : return "setgid";
        #endif
        #ifdef  __NR_getgid
         case __NR_getgid : return "getgid";
        #endif
        #ifdef  __NR_signal
         case __NR_signal : return "signal";
        #endif
        #ifdef  __NR_geteuid
         case __NR_geteuid : return "geteuid";
        #endif
        #ifdef  __NR_getegid
         case __NR_getegid : return "getegid";
        #endif
        #ifdef  __NR_acct
         case __NR_acct : return "acct";
        #endif
        #ifdef  __NR_umount2
         case __NR_umount2 : return "umount2";
        #endif
        #ifdef  __NR_lock
         case __NR_lock : return "lock";
        #endif
        #ifdef  __NR_ioctl
         case __NR_ioctl : return "ioctl";
        #endif
        #ifdef  __NR_fcntl
         case __NR_fcntl : return "fcntl";
        #endif
        #ifdef  __NR_mpx
         case __NR_mpx : return "mpx";
        #endif
        #ifdef  __NR_setpgid
         case __NR_setpgid : return "setpgid";
        #endif
        #ifdef  __NR_ulimit
         case __NR_ulimit : return "ulimit";
        #endif
        #ifdef  __NR_oldolduname
         case __NR_oldolduname : return "oldolduname";
        #endif
        #ifdef  __NR_umask
         case __NR_umask : return "umask";
        #endif
        #ifdef  __NR_chroot
         case __NR_chroot : return "chroot";
        #endif
        #ifdef  __NR_ustat
         case __NR_ustat : return "ustat";
        #endif
        #ifdef  __NR_dup2
         case __NR_dup2 : return "dup2";
        #endif
        #ifdef  __NR_getppid
         case __NR_getppid : return "getppid";
        #endif
        #ifdef  __NR_getpgrp
         case __NR_getpgrp : return "getpgrp";
        #endif
        #ifdef  __NR_setsid
         case __NR_setsid : return "setsid";
        #endif
        #ifdef  __NR_sigaction
         case __NR_sigaction : return "sigaction";
        #endif
        #ifdef  __NR_sgetmask
         case __NR_sgetmask : return "sgetmask";
        #endif
        #ifdef  __NR_ssetmask
         case __NR_ssetmask : return "ssetmask";
        #endif
        #ifdef  __NR_setreuid
         case __NR_setreuid : return "setreuid";
        #endif
        #ifdef  __NR_setregid
         case __NR_setregid : return "setregid";
        #endif
        #ifdef  __NR_sigsuspend
         case __NR_sigsuspend : return "sigsuspend";
        #endif
        #ifdef  __NR_sigpending
         case __NR_sigpending : return "sigpending";
        #endif
        #ifdef  __NR_sethostname
         case __NR_sethostname : return "sethostname";
        #endif
        #ifdef  __NR_setrlimit
         case __NR_setrlimit : return "setrlimit";
        #endif
        #ifdef  __NR_getrlimit
         case __NR_getrlimit : return "getrlimit";
        #endif
        #ifdef  __NR_getrusage
         case __NR_getrusage : return "getrusage";
        #endif
        #ifdef  __NR_gettimeofday
         case __NR_gettimeofday : return "gettimeofday";
        #endif
        #ifdef  __NR_settimeofday
         case __NR_settimeofday : return "settimeofday";
        #endif
        #ifdef  __NR_getgroups
         case __NR_getgroups : return "getgroups";
        #endif
        #ifdef  __NR_setgroups
         case __NR_setgroups : return "setgroups";
        #endif
        #ifdef  __NR_select
         case __NR_select : return "select";
        #endif
        #ifdef  __NR_symlink
         case __NR_symlink : return "symlink";
        #endif
        #ifdef  __NR_oldlstat
         case __NR_oldlstat : return "oldlstat";
        #endif
        #ifdef  __NR_readlink
         case __NR_readlink : return "readlink";
        #endif
        #ifdef  __NR_uselib
         case __NR_uselib : return "uselib";
        #endif
        #ifdef  __NR_swapon
         case __NR_swapon : return "swapon";
        #endif
        #ifdef  __NR_reboot
         case __NR_reboot : return "reboot";
        #endif
        #ifdef  __NR_readdir
         case __NR_readdir : return "readdir";
        #endif
        #ifdef  __NR_mmap
         case __NR_mmap : return "mmap";
        #endif
        #ifdef  __NR_munmap
         case __NR_munmap : return "munmap";
        #endif
        #ifdef  __NR_truncate
         case __NR_truncate : return "truncate";
        #endif
        #ifdef  __NR_ftruncate
         case __NR_ftruncate : return "ftruncate";
        #endif
        #ifdef  __NR_fchmod
         case __NR_fchmod : return "fchmod";
        #endif
        #ifdef  __NR_fchown
         case __NR_fchown : return "fchown";
        #endif
        #ifdef  __NR_getpriority
         case __NR_getpriority : return "getpriority";
        #endif
        #ifdef  __NR_setpriority
         case __NR_setpriority : return "setpriority";
        #endif
        #ifdef  __NR_profil
         case __NR_profil : return "profil";
        #endif
        #ifdef  __NR_statfs
         case __NR_statfs : return "statfs";
        #endif
        #ifdef  __NR_fstatfs
         case __NR_fstatfs : return "fstatfs";
        #endif
        #ifdef  __NR_ioperm
         case __NR_ioperm : return "ioperm";
        #endif
        #ifdef  __NR_socketcall
         case __NR_socketcall : return "socketcall";
        #endif
        #ifdef  __NR_syslog
         case __NR_syslog : return "syslog";
        #endif
        #ifdef  __NR_setitimer
         case __NR_setitimer : return "setitimer";
        #endif
        #ifdef  __NR_getitimer
         case __NR_getitimer : return "getitimer";
        #endif
        #ifdef  __NR_stat
         case __NR_stat : return "stat";
        #endif
        #ifdef  __NR_lstat
         case __NR_lstat : return "lstat";
        #endif
        #ifdef  __NR_fstat
         case __NR_fstat : return "fstat";
        #endif
        #ifdef  __NR_olduname
         case __NR_olduname : return "olduname";
        #endif
        #ifdef  __NR_iopl
         case __NR_iopl : return "iopl";
        #endif
        #ifdef  __NR_vhangup
         case __NR_vhangup : return "vhangup";
        #endif
        #ifdef  __NR_idle
         case __NR_idle : return "idle";
        #endif
        #ifdef  __NR_vm86
         case __NR_vm86 : return "vm86";
        #endif
        #ifdef  __NR_wait4
         case __NR_wait4 : return "wait4";
        #endif
        #ifdef  __NR_swapoff
         case __NR_swapoff : return "swapoff";
        #endif
        #ifdef  __NR_sysinfo
         case __NR_sysinfo : return "sysinfo";
        #endif
        #ifdef  __NR_ipc
         case __NR_ipc : return "ipc";
        #endif
        #ifdef  __NR_fsync
         case __NR_fsync : return "fsync";
        #endif
        #ifdef  __NR_sigreturn
         case __NR_sigreturn : return "sigreturn";
        #endif
        #ifdef  __NR_clone
         case __NR_clone : return "clone";
        #endif
        #ifdef  __NR_setdomainname
         case __NR_setdomainname : return "setdomainname";
        #endif
        #ifdef  __NR_uname
         case __NR_uname : return "uname";
        #endif
        #ifdef  __NR_modify_ldt
         case __NR_modify_ldt : return "modify_ldt";
        #endif
        #ifdef  __NR_adjtimex
         case __NR_adjtimex : return "adjtimex";
        #endif
        #ifdef  __NR_mprotect
         case __NR_mprotect : return "mprotect";
        #endif
        #ifdef  __NR_sigprocmask
         case __NR_sigprocmask : return "sigprocmask";
        #endif
        #ifdef  __NR_create_module
         case __NR_create_module : return "create_module";
        #endif
        #ifdef  __NR_init_module
         case __NR_init_module : return "init_module";
        #endif
        #ifdef  __NR_delete_module
         case __NR_delete_module : return "delete_module";
        #endif
        #ifdef  __NR_get_kernel_syms
         case __NR_get_kernel_syms : return "get_kernel_syms";
        #endif
        #ifdef  __NR_quotactl
         case __NR_quotactl : return "quotactl";
        #endif
        #ifdef  __NR_getpgid
         case __NR_getpgid : return "getpgid";
        #endif
        #ifdef  __NR_fchdir
         case __NR_fchdir : return "fchdir";
        #endif
        #ifdef  __NR_bdflush
         case __NR_bdflush : return "bdflush";
        #endif
        #ifdef  __NR_sysfs
         case __NR_sysfs : return "sysfs";
        #endif
        #ifdef  __NR_personality
         case __NR_personality : return "personality";
        #endif
        #ifdef  __NR_afs_syscall
         case __NR_afs_syscall : return "afs_syscall";
        #endif
        #ifdef  __NR_setfsuid
         case __NR_setfsuid : return "setfsuid";
        #endif
        #ifdef  __NR_setfsgid
         case __NR_setfsgid : return "setfsgid";
        #endif
        #ifdef  __NR__llseek
         case __NR__llseek : return "_llseek";
        #endif
        #ifdef  __NR_getdents
         case __NR_getdents : return "getdents";
        #endif
        #ifdef  __NR__newselect
         case __NR__newselect : return "_newselect";
        #endif
        #ifdef  __NR_flock
         case __NR_flock : return "flock";
        #endif
        #ifdef  __NR_msync
         case __NR_msync : return "msync";
        #endif
        #ifdef  __NR_readv
         case __NR_readv : return "readv";
        #endif
        #ifdef  __NR_writev
         case __NR_writev : return "writev";
        #endif
        #ifdef  __NR_getsid
         case __NR_getsid : return "getsid";
        #endif
        #ifdef  __NR_fdatasync
         case __NR_fdatasync : return "fdatasync";
        #endif
        #ifdef  __NR__sysctl
         case __NR__sysctl : return "_sysctl";
        #endif
        #ifdef  __NR_mlock
         case __NR_mlock : return "mlock";
        #endif
        #ifdef  __NR_munlock
         case __NR_munlock : return "munlock";
        #endif
        #ifdef  __NR_mlockall
         case __NR_mlockall : return "mlockall";
        #endif
        #ifdef  __NR_munlockall
         case __NR_munlockall : return "munlockall";
        #endif
        #ifdef  __NR_sched_setparam
         case __NR_sched_setparam : return "sched_setparam";
        #endif
        #ifdef  __NR_sched_getparam
         case __NR_sched_getparam : return "sched_getparam";
        #endif
        #ifdef  __NR_sched_setscheduler
         case __NR_sched_setscheduler : return "sched_setscheduler";
        #endif
        #ifdef  __NR_sched_getscheduler
         case __NR_sched_getscheduler : return "sched_getscheduler";
        #endif
        #ifdef  __NR_sched_yield
         case __NR_sched_yield : return "sched_yield";
        #endif
        #ifdef  __NR_sched_get_priority_max
         case __NR_sched_get_priority_max : return "sched_get_priority_max";
        #endif
        #ifdef  __NR_sched_get_priority_min
         case __NR_sched_get_priority_min : return "sched_get_priority_min";
        #endif
        #ifdef  __NR_sched_rr_get_interval
         case __NR_sched_rr_get_interval : return "sched_rr_get_interval";
        #endif
        #ifdef  __NR_nanosleep
         case __NR_nanosleep : return "nanosleep";
        #endif
        #ifdef  __NR_mremap
         case __NR_mremap : return "mremap";
        #endif
        #ifdef  __NR_setresuid
         case __NR_setresuid : return "setresuid";
        #endif
        #ifdef  __NR_getresuid
         case __NR_getresuid : return "getresuid";
        #endif
        #ifdef  __NR_query_module
         case __NR_query_module : return "query_module";
        #endif
        #ifdef  __NR_poll
         case __NR_poll : return "poll";
        #endif
        #ifdef  __NR_nfsservctl
         case __NR_nfsservctl : return "nfsservctl";
        #endif
        #ifdef  __NR_setresgid
         case __NR_setresgid : return "setresgid";
        #endif
        #ifdef  __NR_getresgid
         case __NR_getresgid : return "getresgid";
        #endif
        #ifdef  __NR_prctl
         case __NR_prctl : return "prctl";
        #endif
        #ifdef  __NR_rt_sigreturn
         case __NR_rt_sigreturn : return "rt_sigreturn";
        #endif
        #ifdef  __NR_rt_sigaction
         case __NR_rt_sigaction : return "rt_sigaction";
        #endif
        #ifdef  __NR_rt_sigprocmask
         case __NR_rt_sigprocmask : return "rt_sigprocmask";
        #endif
        #ifdef  __NR_rt_sigpending
         case __NR_rt_sigpending : return "rt_sigpending";
        #endif
        #ifdef  __NR_rt_sigtimedwait
         case __NR_rt_sigtimedwait : return "rt_sigtimedwait";
        #endif
        #ifdef  __NR_rt_sigqueueinfo
         case __NR_rt_sigqueueinfo : return "rt_sigqueueinfo";
        #endif
        #ifdef  __NR_rt_sigsuspend
         case __NR_rt_sigsuspend : return "rt_sigsuspend";
        #endif
        #ifdef  __NR_pread64
         case __NR_pread64 : return "pread64";
        #endif
        #ifdef  __NR_pwrite64
         case __NR_pwrite64 : return "pwrite64";
        #endif
        #ifdef  __NR_chown
         case __NR_chown : return "chown";
        #endif
        #ifdef  __NR_getcwd
         case __NR_getcwd : return "getcwd";
        #endif
        #ifdef  __NR_capget
         case __NR_capget : return "capget";
        #endif
        #ifdef  __NR_capset
         case __NR_capset : return "capset";
        #endif
        #ifdef  __NR_sigaltstack
         case __NR_sigaltstack : return "sigaltstack";
        #endif
        #ifdef  __NR_sendfile
         case __NR_sendfile : return "sendfile";
        #endif
        #ifdef  __NR_getpmsg
         case __NR_getpmsg : return "getpmsg";
        #endif
        #ifdef  __NR_putpmsg
         case __NR_putpmsg : return "putpmsg";
        #endif
        #ifdef  __NR_vfork
         case __NR_vfork : return "vfork";
        #endif
        #ifdef  __NR_ugetrlimit
         case __NR_ugetrlimit : return "ugetrlimit";
        #endif
        #ifdef  __NR_readahead
         case __NR_readahead : return "readahead";
        #endif
        #ifdef  __NR_mmap2
         case __NR_mmap2 : return "mmap2";
        #endif
        #ifdef  __NR_truncate64
         case __NR_truncate64 : return "truncate64";
        #endif
        #ifdef  __NR_ftruncate64
         case __NR_ftruncate64 : return "ftruncate64";
        #endif
        #ifdef  __NR_stat64
         case __NR_stat64 : return "stat64";
        #endif
        #ifdef  __NR_lstat64
         case __NR_lstat64 : return "lstat64";
        #endif
        #ifdef  __NR_fstat64
         case __NR_fstat64 : return "fstat64";
        #endif
        #ifdef  __NR_pciconfig_read
         case __NR_pciconfig_read : return "pciconfig_read";
        #endif
        #ifdef  __NR_pciconfig_write
         case __NR_pciconfig_write : return "pciconfig_write";
        #endif
        #ifdef  __NR_pciconfig_iobase
         case __NR_pciconfig_iobase : return "pciconfig_iobase";
        #endif
        #ifdef  __NR_multiplexer
         case __NR_multiplexer : return "multiplexer";
        #endif
        #ifdef  __NR_getdents64
         case __NR_getdents64 : return "getdents64";
        #endif
        #ifdef  __NR_pivot_root
         case __NR_pivot_root : return "pivot_root";
        #endif
        #ifdef  __NR_fcntl64
         case __NR_fcntl64 : return "fcntl64";
        #endif
        #ifdef  __NR_madvise
         case __NR_madvise : return "madvise";
        #endif
        #ifdef  __NR_mincore
         case __NR_mincore : return "mincore";
        #endif
        #ifdef  __NR_gettid
         case __NR_gettid : return "gettid";
        #endif
        #ifdef  __NR_tkill
         case __NR_tkill : return "tkill";
        #endif
        #ifdef  __NR_setxattr
         case __NR_setxattr : return "setxattr";
        #endif
        #ifdef  __NR_lsetxattr
         case __NR_lsetxattr : return "lsetxattr";
        #endif
        #ifdef  __NR_fsetxattr
         case __NR_fsetxattr : return "fsetxattr";
        #endif
        #ifdef  __NR_getxattr
         case __NR_getxattr : return "getxattr";
        #endif
        #ifdef  __NR_lgetxattr
         case __NR_lgetxattr : return "lgetxattr";
        #endif
        #ifdef  __NR_fgetxattr
         case __NR_fgetxattr : return "fgetxattr";
        #endif
        #ifdef  __NR_listxattr
         case __NR_listxattr : return "listxattr";
        #endif
        #ifdef  __NR_llistxattr
         case __NR_llistxattr : return "llistxattr";
        #endif
        #ifdef  __NR_flistxattr
         case __NR_flistxattr : return "flistxattr";
        #endif
        #ifdef  __NR_removexattr
         case __NR_removexattr : return "removexattr";
        #endif
        #ifdef  __NR_lremovexattr
         case __NR_lremovexattr : return "lremovexattr";
        #endif
        #ifdef  __NR_fremovexattr
         case __NR_fremovexattr : return "fremovexattr";
        #endif
        #ifdef  __NR_futex
         case __NR_futex : return "futex";
        #endif
        #ifdef  __NR_sched_setaffinity
         case __NR_sched_setaffinity : return "sched_setaffinity";
        #endif
        #ifdef  __NR_sched_getaffinity
         case __NR_sched_getaffinity : return "sched_getaffinity";
        #endif
        #ifdef  __NR_tuxcall
         case __NR_tuxcall : return "tuxcall";
        #endif
        #ifdef  __NR_sendfile64
         case __NR_sendfile64 : return "sendfile64";
        #endif
        #ifdef  __NR_io_setup
         case __NR_io_setup : return "io_setup";
        #endif
        #ifdef  __NR_io_destroy
         case __NR_io_destroy : return "io_destroy";
        #endif
        #ifdef  __NR_io_getevents
         case __NR_io_getevents : return "io_getevents";
        #endif
        #ifdef  __NR_io_submit
         case __NR_io_submit : return "io_submit";
        #endif
        #ifdef  __NR_io_cancel
         case __NR_io_cancel : return "io_cancel";
        #endif
        #ifdef  __NR_set_tid_address
         case __NR_set_tid_address : return "set_tid_address";
        #endif
        #ifdef  __NR_fadvise64
         case __NR_fadvise64 : return "fadvise64";
        #endif
        #ifdef  __NR_exit_group
         case __NR_exit_group : return "exit_group";
        #endif
        #ifdef  __NR_lookup_dcookie
         case __NR_lookup_dcookie : return "lookup_dcookie";
        #endif
        #ifdef  __NR_epoll_create
         case __NR_epoll_create : return "epoll_create";
        #endif
        #ifdef  __NR_epoll_ctl
         case __NR_epoll_ctl : return "epoll_ctl";
        #endif
        #ifdef  __NR_epoll_wait
         case __NR_epoll_wait : return "epoll_wait";
        #endif
        #ifdef  __NR_remap_file_pages
         case __NR_remap_file_pages : return "remap_file_pages";
        #endif
        #ifdef  __NR_timer_create
         case __NR_timer_create : return "timer_create";
        #endif
        #ifdef  __NR_timer_settime
         case __NR_timer_settime : return "timer_settime";
        #endif
        #ifdef  __NR_timer_gettime
         case __NR_timer_gettime : return "timer_gettime";
        #endif
        #ifdef  __NR_timer_getoverrun
         case __NR_timer_getoverrun : return "timer_getoverrun";
        #endif
        #ifdef  __NR_timer_delete
         case __NR_timer_delete : return "timer_delete";
        #endif
        #ifdef  __NR_clock_settime
         case __NR_clock_settime : return "clock_settime";
        #endif
        #ifdef  __NR_clock_gettime
         case __NR_clock_gettime : return "clock_gettime";
        #endif
        #ifdef  __NR_clock_getres
         case __NR_clock_getres : return "clock_getres";
        #endif
        #ifdef  __NR_clock_nanosleep
         case __NR_clock_nanosleep : return "clock_nanosleep";
        #endif
        #ifdef  __NR_swapcontext
         case __NR_swapcontext : return "swapcontext";
        #endif
        #ifdef  __NR_tgkill
         case __NR_tgkill : return "tgkill";
        #endif
        #ifdef  __NR_utimes
         case __NR_utimes : return "utimes";
        #endif
        #ifdef  __NR_statfs64
         case __NR_statfs64 : return "statfs64";
        #endif
        #ifdef  __NR_fstatfs64
         case __NR_fstatfs64 : return "fstatfs64";
        #endif
        #ifdef  __NR_fadvise64_64
         case __NR_fadvise64_64 : return "fadvise64_64";
        #endif
        #ifdef  __NR_rtas
         case __NR_rtas : return "rtas";
        #endif
        #ifdef  __NR_sys_debug_setcontext
         case __NR_sys_debug_setcontext : return "sys_debug_setcontext";
        #endif
        #ifdef  __NR_migrate_pages
         case __NR_migrate_pages : return "migrate_pages";
        #endif
        #ifdef  __NR_mbind
         case __NR_mbind : return "mbind";
        #endif
        #ifdef  __NR_get_mempolicy
         case __NR_get_mempolicy : return "get_mempolicy";
        #endif
        #ifdef  __NR_set_mempolicy
         case __NR_set_mempolicy : return "set_mempolicy";
        #endif
        #ifdef  __NR_mq_open
         case __NR_mq_open : return "mq_open";
        #endif
        #ifdef  __NR_mq_unlink
         case __NR_mq_unlink : return "mq_unlink";
        #endif
        #ifdef  __NR_mq_timedsend
         case __NR_mq_timedsend : return "mq_timedsend";
        #endif
        #ifdef  __NR_mq_timedreceive
         case __NR_mq_timedreceive : return "mq_timedreceive";
        #endif
        #ifdef  __NR_mq_notify
         case __NR_mq_notify : return "mq_notify";
        #endif
        #ifdef  __NR_mq_getsetattr
         case __NR_mq_getsetattr : return "mq_getsetattr";
        #endif
        #ifdef  __NR_kexec_load
         case __NR_kexec_load : return "kexec_load";
        #endif
        #ifdef  __NR_add_key
         case __NR_add_key : return "add_key";
        #endif
        #ifdef  __NR_request_key
         case __NR_request_key : return "request_key";
        #endif
        #ifdef  __NR_keyctl
         case __NR_keyctl : return "keyctl";
        #endif
        #ifdef  __NR_waitid
         case __NR_waitid : return "waitid";
        #endif
        #ifdef  __NR_ioprio_set
         case __NR_ioprio_set : return "ioprio_set";
        #endif
        #ifdef  __NR_ioprio_get
         case __NR_ioprio_get : return "ioprio_get";
        #endif
        #ifdef  __NR_inotify_init
         case __NR_inotify_init : return "inotify_init";
        #endif
        #ifdef  __NR_inotify_add_watch
         case __NR_inotify_add_watch : return "inotify_add_watch";
        #endif
        #ifdef  __NR_inotify_rm_watch
         case __NR_inotify_rm_watch : return "inotify_rm_watch";
        #endif
        #ifdef  __NR_spu_run
         case __NR_spu_run : return "spu_run";
        #endif
        #ifdef  __NR_spu_create
         case __NR_spu_create : return "spu_create";
        #endif
        #ifdef  __NR_pselect6
         case __NR_pselect6 : return "pselect6";
        #endif
        #ifdef  __NR_ppoll
         case __NR_ppoll : return "ppoll";
        #endif
        #ifdef  __NR_unshare
         case __NR_unshare : return "unshare";
        #endif
        #ifdef  __NR_splice
         case __NR_splice : return "splice";
        #endif
        #ifdef  __NR_tee
         case __NR_tee : return "tee";
        #endif
        #ifdef  __NR_vmsplice
         case __NR_vmsplice : return "vmsplice";
        #endif
        #ifdef  __NR_openat
         case __NR_openat : return "openat";
        #endif
        #ifdef  __NR_mkdirat
         case __NR_mkdirat : return "mkdirat";
        #endif
        #ifdef  __NR_mknodat
         case __NR_mknodat : return "mknodat";
        #endif
        #ifdef  __NR_fchownat
         case __NR_fchownat : return "fchownat";
        #endif
        #ifdef  __NR_futimesat
         case __NR_futimesat : return "futimesat";
        #endif
        #ifdef  __NR_newfstatat
         case __NR_newfstatat : return "newfstatat";
        #endif
        #ifdef  __NR_fstatat64
         case __NR_fstatat64 : return "fstatat64";
        #endif
        #ifdef  __NR_unlinkat
         case __NR_unlinkat : return "unlinkat";
        #endif
        #ifdef  __NR_renameat
         case __NR_renameat : return "renameat";
        #endif
        #ifdef  __NR_linkat
         case __NR_linkat : return "linkat";
        #endif
        #ifdef  __NR_symlinkat
         case __NR_symlinkat : return "symlinkat";
        #endif
        #ifdef  __NR_readlinkat
         case __NR_readlinkat : return "readlinkat";
        #endif
        #ifdef  __NR_fchmodat
         case __NR_fchmodat : return "fchmodat";
        #endif
        #ifdef  __NR_faccessat
         case __NR_faccessat : return "faccessat";
        #endif
        #ifdef  __NR_get_robust_list
         case __NR_get_robust_list : return "get_robust_list";
        #endif
        #ifdef  __NR_set_robust_list
         case __NR_set_robust_list : return "set_robust_list";
        #endif
        #ifdef  __NR_move_pages
         case __NR_move_pages : return "move_pages";
        #endif
        #ifdef  __NR_getcpu
         case __NR_getcpu : return "getcpu";
        #endif
        #ifdef  __NR_epoll_pwait
         case __NR_epoll_pwait : return "epoll_pwait";
        #endif
        #ifdef  __NR_utimensat
         case __NR_utimensat : return "utimensat";
        #endif
        #ifdef  __NR_signalfd
         case __NR_signalfd : return "signalfd";
        #endif
        #ifdef  __NR_timerfd_create
         case __NR_timerfd_create : return "timerfd_create";
        #endif
        #ifdef  __NR_eventfd
         case __NR_eventfd : return "eventfd";
        #endif
        #ifdef  __NR_sync_file_range2
         case __NR_sync_file_range2 : return "sync_file_range2";
        #endif
        #ifdef  __NR_fallocate
         case __NR_fallocate : return "fallocate";
        #endif
        #ifdef  __NR_subpage_prot
         case __NR_subpage_prot : return "subpage_prot";
        #endif
        #ifdef  __NR_timerfd_settime
         case __NR_timerfd_settime : return "timerfd_settime";
        #endif
        #ifdef  __NR_timerfd_gettime
         case __NR_timerfd_gettime : return "timerfd_gettime";
        #endif
        #ifdef  __NR_signalfd4
         case __NR_signalfd4 : return "signalfd4";
        #endif
        #ifdef  __NR_eventfd2
         case __NR_eventfd2 : return "eventfd2";
        #endif
        #ifdef  __NR_epoll_create1
         case __NR_epoll_create1 : return "epoll_create1";
        #endif
        #ifdef  __NR_dup3
         case __NR_dup3 : return "dup3";
        #endif
        #ifdef  __NR_pipe2
         case __NR_pipe2 : return "pipe2";
        #endif
        #ifdef  __NR_inotify_init1
         case __NR_inotify_init1 : return "inotify_init1";
        #endif
        #ifdef  __NR_perf_event_open
         case __NR_perf_event_open : return "perf_event_open";
        #endif
        #ifdef  __NR_preadv
         case __NR_preadv : return "preadv";
        #endif
        #ifdef  __NR_pwritev
         case __NR_pwritev : return "pwritev";
        #endif
        #ifdef  __NR_rt_tgsigqueueinfo
         case __NR_rt_tgsigqueueinfo : return "rt_tgsigqueueinfo";
        #endif
        #ifdef  __NR_fanotify_init
         case __NR_fanotify_init : return "fanotify_init";
        #endif
        #ifdef  __NR_fanotify_mark
         case __NR_fanotify_mark : return "fanotify_mark";
        #endif
        #ifdef  __NR_prlimit64
         case __NR_prlimit64 : return "prlimit64";
        #endif
        #ifdef  __NR_socket
         case __NR_socket : return "socket";
        #endif
        #ifdef  __NR_bind
         case __NR_bind : return "bind";
        #endif
        #ifdef  __NR_connect
         case __NR_connect : return "connect";
        #endif
        #ifdef  __NR_listen
         case __NR_listen : return "listen";
        #endif
        #ifdef  __NR_accept
         case __NR_accept : return "accept";
        #endif
        #ifdef  __NR_getsockname
         case __NR_getsockname : return "getsockname";
        #endif
        #ifdef  __NR_getpeername
         case __NR_getpeername : return "getpeername";
        #endif
        #ifdef  __NR_socketpair
         case __NR_socketpair : return "socketpair";
        #endif
        #ifdef  __NR_send
         case __NR_send : return "send";
        #endif
        #ifdef  __NR_sendto
         case __NR_sendto : return "sendto";
        #endif
        #ifdef  __NR_recv
         case __NR_recv : return "recv";
        #endif
        #ifdef  __NR_recvfrom
         case __NR_recvfrom : return "recvfrom";
        #endif
        #ifdef  __NR_shutdown
         case __NR_shutdown : return "shutdown";
        #endif
        #ifdef  __NR_setsockopt
         case __NR_setsockopt : return "setsockopt";
        #endif
        #ifdef  __NR_getsockopt
         case __NR_getsockopt : return "getsockopt";
        #endif
        #ifdef  __NR_sendmsg
         case __NR_sendmsg : return "sendmsg";
        #endif
        #ifdef  __NR_recvmsg
         case __NR_recvmsg : return "recvmsg";
        #endif
        #ifdef  __NR_recvmmsg
         case __NR_recvmmsg : return "recvmmsg";
        #endif
        #ifdef  __NR_accept4
         case __NR_accept4 : return "accept4";
        #endif
        #ifdef  __NR_name_to_handle_at
         case __NR_name_to_handle_at : return "name_to_handle_at";
        #endif
        #ifdef  __NR_open_by_handle_at
         case __NR_open_by_handle_at : return "open_by_handle_at";
        #endif
        #ifdef  __NR_clock_adjtime
         case __NR_clock_adjtime : return "clock_adjtime";
        #endif
        #ifdef  __NR_syncfs
         case __NR_syncfs : return "syncfs";
        #endif
        #ifdef  __NR_sendmmsg
         case __NR_sendmmsg : return "sendmmsg";
        #endif
        #ifdef  __NR_setns
         case __NR_setns : return "setns";
        #endif
        #ifdef  __NR_process_vm_readv
         case __NR_process_vm_readv : return "process_vm_readv";
        #endif
        #ifdef  __NR_process_vm_writev
         case __NR_process_vm_writev : return "process_vm_writev";
        #endif
        #ifdef  __NR_finit_module
         case __NR_finit_module : return "finit_module";
        #endif
        #ifdef  __NR_kcmp
         case __NR_kcmp : return "kcmp";
        #endif
        #ifdef  __NR_sched_setattr
         case __NR_sched_setattr : return "sched_setattr";
        #endif
        #ifdef  __NR_sched_getattr
         case __NR_sched_getattr : return "sched_getattr";
        #endif
        #ifdef  __NR_renameat2
         case __NR_renameat2 : return "renameat2";
        #endif
        #ifdef  __NR_seccomp
         case __NR_seccomp : return "seccomp";
        #endif
        #ifdef  __NR_getrandom
         case __NR_getrandom : return "getrandom";
        #endif
        #ifdef  __NR_memfd_create
         case __NR_memfd_create : return "memfd_create";
        #endif
        #ifdef  __NR_bpf
         case __NR_bpf : return "bpf";
        #endif
        #ifdef  __NR_execveat
         case __NR_execveat : return "execveat";
        #endif
        #ifdef  __NR_switch_endian
         case __NR_switch_endian : return "switch_endian";
        #endif
        #ifdef  __NR_userfaultfd
         case __NR_userfaultfd : return "userfaultfd";
        #endif
        #ifdef  __NR_membarrier
         case __NR_membarrier : return "membarrier";
        #endif
        #ifdef  __NR_mlock2
         case __NR_mlock2 : return "mlock2";
        #endif
        #ifdef  __NR_copy_file_range
         case __NR_copy_file_range : return "copy_file_range";
        #endif
        #ifdef  __NR_preadv2
         case __NR_preadv2 : return "preadv2";
        #endif
        #ifdef  __NR_pwritev2
         case __NR_pwritev2 : return "pwritev2";
        #endif
        #ifdef  __NR_kexec_file_load
         case __NR_kexec_file_load : return "kexec_file_load";
        #endif
        #ifdef  __NR_statx
         case __NR_statx : return "statx";
        #endif
        #ifdef  __NR_pkey_alloc
         case __NR_pkey_alloc : return "pkey_alloc";
        #endif
        #ifdef  __NR_pkey_free
         case __NR_pkey_free : return "pkey_free";
        #endif
        #ifdef  __NR_pkey_mprotect
         case __NR_pkey_mprotect : return "pkey_mprotect";
        #endif
        #ifdef  __NR_rseq
         case __NR_rseq : return "rseq";
        #endif
        #ifdef  __NR_io_pgetevents
         case __NR_io_pgetevents : return "io_pgetevents";
        #endif
        #ifdef  __NR_pidfd_send_signal
         case __NR_pidfd_send_signal : return "pidfd_send_signal";
        #endif
        #ifdef  __NR_io_uring_setup
         case __NR_io_uring_setup : return "io_uring_setup";
        #endif
        #ifdef  __NR_io_uring_enter
         case __NR_io_uring_enter : return "io_uring_enter";
        #endif
        #ifdef  __NR_io_uring_register
         case __NR_io_uring_register : return "io_uring_register";
        #endif
        #ifdef  __NR_open_tree
         case __NR_open_tree : return "open_tree";
        #endif
        #ifdef  __NR_move_mount
         case __NR_move_mount : return "move_mount";
        #endif
        #ifdef  __NR_fsopen
         case __NR_fsopen : return "fsopen";
        #endif
        #ifdef  __NR_fsconfig
         case __NR_fsconfig : return "fsconfig";
        #endif
        #ifdef  __NR_fsmount
         case __NR_fsmount : return "fsmount";
        #endif
        #ifdef  __NR_fspick
         case __NR_fspick : return "fspick";
        #endif
        #ifdef  __NR_close_range
         case __NR_close_range : return "close_range";
        #endif
        #ifdef  __NR_openat2
         case __NR_openat2 : return "openat2";
        #endif
        #ifdef  __NR_faccessat2
         case __NR_faccessat2 : return "faccessat2";
        #endif



        default:
            return "unknown_syscall";

    }
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
