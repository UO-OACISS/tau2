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
    __scalls[288] = "accept4";
    __scalls[43] = "accept";
    __scalls[21] = "access";
    __scalls[163] = "acct";
    __scalls[248] = "add_key";
    __scalls[159] = "adjtimex";
    __scalls[37] = "alarm";
    __scalls[158] = "arch_prctl";
    __scalls[49] = "bind";
    __scalls[321] = "bpf";
    __scalls[12] = "brk";
    __scalls[125] = "capget";
    __scalls[126] = "capset";
    __scalls[80] = "chdir";
    __scalls[90] = "chmod";
    __scalls[92] = "chown";
    __scalls[161] = "chroot";
    __scalls[305] = "clock_adjtime";
    __scalls[229] = "clock_getres";
    __scalls[228] = "clock_gettime";
    __scalls[230] = "clock_nanosleep";
    __scalls[227] = "clock_settime";
    __scalls[56] = "clone";
    __scalls[3] = "close";
    __scalls[42] = "connect";
    __scalls[326] = "copy_file_range";
    __scalls[85] = "creat";
    __scalls[176] = "delete_module";
    __scalls[33] = "dup2";
    __scalls[292] = "dup3";
    __scalls[32] = "dup";
    __scalls[291] = "epoll_create1";
    __scalls[213] = "epoll_create";
    __scalls[233] = "epoll_ctl";
    __scalls[281] = "epoll_pwait";
    __scalls[232] = "epoll_wait";
    __scalls[290] = "eventfd2";
    __scalls[284] = "eventfd";
    __scalls[322] = "execveat";
    __scalls[59] = "execve";
    __scalls[60] = "exit";
    __scalls[231] = "exit_group";
    __scalls[269] = "faccessat";
    __scalls[221] = "fadvise64";
    __scalls[285] = "fallocate";
    __scalls[300] = "fanotify_init";
    __scalls[301] = "fanotify_mark";
    __scalls[81] = "fchdir";
    __scalls[268] = "fchmodat";
    __scalls[91] = "fchmod";
    __scalls[260] = "fchownat";
    __scalls[93] = "fchown";
    __scalls[72] = "fcntl";
    __scalls[75] = "fdatasync";
    __scalls[193] = "fgetxattr";
    __scalls[313] = "finit_module";
    __scalls[196] = "flistxattr";
    __scalls[73] = "flock";
    __scalls[57] = "fork";
    __scalls[199] = "fremovexattr";
    __scalls[190] = "fsetxattr";
    __scalls[138] = "fstatfs";
    __scalls[5] = "fstat";
    __scalls[74] = "fsync";
    __scalls[77] = "ftruncate";
    __scalls[202] = "futex";
    __scalls[261] = "futimesat";
    __scalls[309] = "getcpu";
    __scalls[79] = "getcwd";
    __scalls[217] = "getdents64";
    __scalls[78] = "getdents";
    __scalls[108] = "getegid";
    __scalls[107] = "geteuid";
    __scalls[104] = "getgid";
    __scalls[115] = "getgroups";
    __scalls[36] = "getitimer";
    __scalls[239] = "get_mempolicy";
    __scalls[52] = "getpeername";
    __scalls[121] = "getpgid";
    __scalls[111] = "getpgrp";
    __scalls[39] = "getpid";
    __scalls[110] = "getppid";
    __scalls[140] = "getpriority";
    __scalls[318] = "getrandom";
    __scalls[120] = "getresgid";
    __scalls[118] = "getresuid";
    __scalls[97] = "getrlimit";
    __scalls[274] = "get_robust_list";
    __scalls[98] = "getrusage";
    __scalls[124] = "getsid";
    __scalls[51] = "getsockname";
    __scalls[55] = "getsockopt";
    __scalls[211] = "get_thread_area";
    __scalls[186] = "gettid";
    __scalls[96] = "gettimeofday";
    __scalls[102] = "getuid";
    __scalls[191] = "getxattr";
    __scalls[175] = "init_module";
    __scalls[254] = "inotify_add_watch";
    __scalls[294] = "inotify_init1";
    __scalls[253] = "inotify_init";
    __scalls[255] = "inotify_rm_watch";
    __scalls[210] = "io_cancel";
    __scalls[16] = "ioctl";
    __scalls[207] = "io_destroy";
    __scalls[208] = "io_getevents";
    __scalls[173] = "ioperm";
    __scalls[333] = "io_pgetevents";
    __scalls[172] = "iopl";
    __scalls[252] = "ioprio_get";
    __scalls[251] = "ioprio_set";
    __scalls[206] = "io_setup";
    __scalls[209] = "io_submit";
    __scalls[312] = "kcmp";
    __scalls[320] = "kexec_file_load";
    __scalls[246] = "kexec_load";
    __scalls[250] = "keyctl";
    __scalls[62] = "kill";
    __scalls[94] = "lchown";
    __scalls[192] = "lgetxattr";
    __scalls[265] = "linkat";
    __scalls[86] = "link";
    __scalls[50] = "listen";
    __scalls[194] = "listxattr";
    __scalls[195] = "llistxattr";
    __scalls[212] = "lookup_dcookie";
    __scalls[198] = "lremovexattr";
    __scalls[8] = "lseek";
    __scalls[189] = "lsetxattr";
    __scalls[6] = "lstat";
    __scalls[28] = "madvise";
    __scalls[237] = "mbind";
    __scalls[324] = "membarrier";
    __scalls[319] = "memfd_create";
    __scalls[256] = "migrate_pages";
    __scalls[27] = "mincore";
    __scalls[258] = "mkdirat";
    __scalls[83] = "mkdir";
    __scalls[259] = "mknodat";
    __scalls[133] = "mknod";
    __scalls[325] = "mlock2";
    __scalls[151] = "mlockall";
    __scalls[149] = "mlock";
    __scalls[9] = "mmap";
    __scalls[154] = "modify_ldt";
    __scalls[165] = "mount";
    __scalls[279] = "move_pages";
    __scalls[10] = "mprotect";
    __scalls[245] = "mq_getsetattr";
    __scalls[244] = "mq_notify";
    __scalls[240] = "mq_open";
    __scalls[243] = "mq_timedreceive";
    __scalls[242] = "mq_timedsend";
    __scalls[241] = "mq_unlink";
    __scalls[25] = "mremap";
    __scalls[71] = "msgctl";
    __scalls[68] = "msgget";
    __scalls[70] = "msgrcv";
    __scalls[69] = "msgsnd";
    __scalls[26] = "msync";
    __scalls[152] = "munlockall";
    __scalls[150] = "munlock";
    __scalls[11] = "munmap";
    __scalls[303] = "name_to_handle_at";
    __scalls[35] = "nanosleep";
    __scalls[262] = "newfstatat";
    __scalls[257] = "openat";
    __scalls[304] = "open_by_handle_at";
    __scalls[2] = "open";
    __scalls[34] = "pause";
    __scalls[298] = "perf_event_open";
    __scalls[135] = "personality";
    __scalls[293] = "pipe2";
    __scalls[22] = "pipe";
    __scalls[155] = "pivot_root";
    __scalls[330] = "pkey_alloc";
    __scalls[331] = "pkey_free";
    __scalls[329] = "pkey_mprotect";
    __scalls[7] = "poll";
    __scalls[271] = "ppoll";
    __scalls[157] = "prctl";
    __scalls[17] = "pread64";
    __scalls[327] = "preadv2";
    __scalls[295] = "preadv";
    __scalls[302] = "prlimit64";
    __scalls[310] = "process_vm_readv";
    __scalls[311] = "process_vm_writev";
    __scalls[270] = "pselect6";
    __scalls[101] = "ptrace";
    __scalls[18] = "pwrite64";
    __scalls[328] = "pwritev2";
    __scalls[296] = "pwritev";
    __scalls[179] = "quotactl";
    __scalls[187] = "readahead";
    __scalls[267] = "readlinkat";
    __scalls[89] = "readlink";
    __scalls[0] = "read";
    __scalls[19] = "readv";
    __scalls[169] = "reboot";
    __scalls[45] = "recvfrom";
    __scalls[299] = "recvmmsg";
    __scalls[47] = "recvmsg";
    __scalls[216] = "remap_file_pages";
    __scalls[197] = "removexattr";
    __scalls[316] = "renameat2";
    __scalls[264] = "renameat";
    __scalls[82] = "rename";
    __scalls[249] = "request_key";
    __scalls[219] = "restart_syscall";
    __scalls[84] = "rmdir";
    __scalls[334] = "rseq";
    __scalls[13] = "rt_sigaction";
    __scalls[127] = "rt_sigpending";
    __scalls[14] = "rt_sigprocmask";
    __scalls[129] = "rt_sigqueueinfo";
    __scalls[15] = "rt_sigreturn";
    __scalls[130] = "rt_sigsuspend";
    __scalls[128] = "rt_sigtimedwait";
    __scalls[297] = "rt_tgsigqueueinfo";
    __scalls[204] = "sched_getaffinity";
    __scalls[315] = "sched_getattr";
    __scalls[143] = "sched_getparam";
    __scalls[146] = "sched_get_priority_max";
    __scalls[147] = "sched_get_priority_min";
    __scalls[145] = "sched_getscheduler";
    __scalls[148] = "sched_rr_get_interval";
    __scalls[203] = "sched_setaffinity";
    __scalls[314] = "sched_setattr";
    __scalls[142] = "sched_setparam";
    __scalls[144] = "sched_setscheduler";
    __scalls[24] = "sched_yield";
    __scalls[317] = "seccomp";
    __scalls[23] = "select";
    __scalls[66] = "semctl";
    __scalls[64] = "semget";
    __scalls[65] = "semop";
    __scalls[220] = "semtimedop";
    __scalls[40] = "sendfile";
    __scalls[307] = "sendmmsg";
    __scalls[46] = "sendmsg";
    __scalls[44] = "sendto";
    __scalls[171] = "setdomainname";
    __scalls[123] = "setfsgid";
    __scalls[122] = "setfsuid";
    __scalls[106] = "setgid";
    __scalls[116] = "setgroups";
    __scalls[170] = "sethostname";
    __scalls[38] = "setitimer";
    __scalls[238] = "set_mempolicy";
    __scalls[308] = "setns";
    __scalls[109] = "setpgid";
    __scalls[141] = "setpriority";
    __scalls[114] = "setregid";
    __scalls[119] = "setresgid";
    __scalls[117] = "setresuid";
    __scalls[113] = "setreuid";
    __scalls[160] = "setrlimit";
    __scalls[273] = "set_robust_list";
    __scalls[112] = "setsid";
    __scalls[54] = "setsockopt";
    __scalls[205] = "set_thread_area";
    __scalls[218] = "set_tid_address";
    __scalls[164] = "settimeofday";
    __scalls[105] = "setuid";
    __scalls[188] = "setxattr";
    __scalls[30] = "shmat";
    __scalls[31] = "shmctl";
    __scalls[67] = "shmdt";
    __scalls[29] = "shmget";
    __scalls[48] = "shutdown";
    __scalls[131] = "sigaltstack";
    __scalls[289] = "signalfd4";
    __scalls[282] = "signalfd";
    __scalls[53] = "socketpair";
    __scalls[41] = "socket";
    __scalls[275] = "splice";
    __scalls[137] = "statfs";
    __scalls[4] = "stat";
    __scalls[332] = "statx";
    __scalls[168] = "swapoff";
    __scalls[167] = "swapon";
    __scalls[266] = "symlinkat";
    __scalls[88] = "symlink";
    __scalls[277] = "sync_file_range";
    __scalls[306] = "syncfs";
    __scalls[162] = "sync";
    __scalls[139] = "sysfs";
    __scalls[99] = "sysinfo";
    __scalls[103] = "syslog";
    __scalls[276] = "tee";
    __scalls[234] = "tgkill";
    __scalls[222] = "timer_create";
    __scalls[226] = "timer_delete";
    __scalls[283] = "timerfd_create";
    __scalls[287] = "timerfd_gettime";
    __scalls[286] = "timerfd_settime";
    __scalls[225] = "timer_getoverrun";
    __scalls[224] = "timer_gettime";
    __scalls[223] = "timer_settime";
    __scalls[100] = "times";
    __scalls[201] = "time";
    __scalls[200] = "tkill";
    __scalls[76] = "truncate";
    __scalls[95] = "umask";
    __scalls[22] = "umount";
    __scalls[63] = "uname";
    __scalls[263] = "unlinkat";
    __scalls[87] = "unlink";
    __scalls[272] = "unshare";
    __scalls[134] = "uselib";
    __scalls[323] = "userfaultfd";
    __scalls[136] = "ustat";
    __scalls[280] = "utimensat";
    __scalls[235] = "utimes";
    __scalls[132] = "utime";
    __scalls[58] = "vfork";
    __scalls[153] = "vhangup";
    __scalls[278] = "vmsplice";
    __scalls[61] = "wait4";
    __scalls[247] = "waitid";
    __scalls[7] = "waitpid";
    __scalls[20] = "writev";
    __scalls[1] = "write";
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
