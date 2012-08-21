include ${TAU_MAKEFILE} 
CC=$(TAU_UPCC) 
CFLAGS=$(TAU_DEFS)  $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -I..
EXTRA_FLAGS=

AR=$(TAU_AR)
ARFLAGS=rcv 

libtau_gupc_wrap.a: tau_gupc_wrap.o 
	$(AR) $(ARFLAGS) $@ $< 

tau_gupc_wrap.o: wr.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@
clean:
	/bin/rm -f tau_gupc_wrap.o libtau_gupc_wrap.a

