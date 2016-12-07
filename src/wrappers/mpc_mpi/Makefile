include ../../../include/Makefile
include ../Makefile

OBJS=mpc_mpi_wrap.o gen_prof.o
#OBJS=mpc_mpi_wrap.o gen_trace.o

CC=$(TAU_CC) 
CXX=$(TAU_CXX) 
CFLAGS=$(TAU_DEFS)  $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -I. -I.. -I$(TAUROOT)/include/Profile
EXTRA_FLAGS=

AR=$(TAU_AR)
ARFLAGS=rcv 

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: install-static-mpc
	mkdir -p $(INSTALLDEST)/wrappers/mpc_mpi
	cp link_options.tau $(INSTALLDEST)/wrappers/mpc_mpi

install-static-mpc: libmpc_mpi_wrap.a
	$(TAU_STATIC_INSTALL_CMD)

#libmpc_mpi_wrap.a: $(OBJS)
#	$(AR) $(ARFLAGS) $@ $< 

libmpc_mpi_wrap.a: $(OBJS)
	$(AR) $(ARFLAGS) $@ $(OBJS)

mpc_mpi_wrap.o: wr.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@

gen_prof.o: gen_prof.cpp
	$(CXX) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@

gen_trace.o: gen_trace.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@
clean:
	/bin/rm -f mpc_mpi_wrap.o gen_prof.o libmpc_mpi_wrap.a

