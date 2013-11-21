include ../../../include/Makefile
include ../Makefile

CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS)  $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -I..
EXTRA_FLAGS=

AR=$(TAU_AR)
ARFLAGS=rcv 

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: install-static-mpc
	mkdir -p $(INSTALLDEST)/wrappers/mpc_mpi
	cp link_options.tau $(INSTALLDEST)/wrappers/mpc_mpi

install-static-mpc: libmpc_mpi_wrap.a
	$(TAU_STATIC_INSTALL_CMD)

libmpc_mpi_wrap.a: mpc_mpi_wrap.o 
	$(AR) $(ARFLAGS) $@ $< 

mpc_mpi_wrap.o: wr.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@
clean:
	/bin/rm -f mpc_mpi_wrap.o libmpc_mpi_wrap.a

