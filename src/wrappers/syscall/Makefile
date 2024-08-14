include ../../../include/Makefile
include ../Makefile

CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAGS) $(TAU_INCLUDE) $(TAU_MPI_INCLUDE) $(USER_OPT) -I./  #-DDEBUG_PTRACE

OBJS=scalls.o tracee.o ptrace_syscall.o
TARGETS=$(TAU_SYSCALL_SHARED)

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: $(TARGETS)
	$(TAU_SHARED_INSTALL_CMD)

$(TAU_SYSCALL_SHARED): $(OBJS)
	$(TAU_CC) $(TAU_SHFLAGS) $@ $^ $(TAU_LIBDL)

%.o: %.c
	$(TAU_CC) $(CFLAGS) $< -c -o $@

clean:
	$(RM) -f $(OBJS) $(TARGETS)