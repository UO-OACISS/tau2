include ${TAU_MAKEFILE} 
CC=$(TAU_UPCC) 
ARFLAGS=rcv 
CFLAGS=$(TAU_DEFS)  $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -I..
WRAPPER_FLAGS=

libtau_gupc_wrap.a: tau_gupc_wrap.o 
	$(TAU_AR) $(ARFLAGS) $@ $< 

tau_gupc_wrap.o: wr.c
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	/bin/rm -f tau_gupc_wrap.o libtau_gupc_wrap.a

