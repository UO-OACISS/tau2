#ifdef __x86_64__
#include "scalls_x86_64.c"
#elif defined __PPC__
#include "scalls_ppc.c"
#elif defined __aarch64__
#include "scalls_aarch64.c"
#else
// Not implemented
#include <scalls.h>

void scalls_init(void)
{
}

const char *get_syscall_name(int id)
{
    return "UNKNOWN";
}

int get_syscall_id(pid_t pid)
{
    return -1;
}
#endif