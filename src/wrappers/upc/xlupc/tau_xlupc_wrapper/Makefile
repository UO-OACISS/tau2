include ${TAU_MAKEFILE} 
CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS)  $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -I..
EXTRA_FLAGS=

AR=$(TAU_AR)
ARFLAGS=rcv 

libtau_xlupc_wrap.a: tau_xlupc_wrap.o 
	$(AR) $(ARFLAGS) $@ $< 

tau_xlupc_wrap.o: wr.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@
clean:
	/bin/rm -f tau_xlupc_wrap.o libtau_xlupc_wrap.a

