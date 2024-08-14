#ifndef SCALLS_H
#define SCALLS_H

#include <sys/types.h>

void scalls_init(void);

const char *get_syscall_name(int id);

int get_syscall_id(pid_t pid);

#endif /* SCALLS_H */
