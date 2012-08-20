include ${TAU_MAKEFILE} 
CC=$(TAU_CC) 
ARFLAGS=rcv 
CFLAGS=$(TAU_DEFS)  $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -I.. 
EXTRA_FLAGS=

libtau_upcr_wrap.a: tau_upcr_wrap.o 
	$(TAU_AR) $(ARFLAGS) $@ $< 

tau_upcr_wrap.o: wr.c
	$(CC) $(CFLAGS) $(WRAPPER_FLAGS) -c $< -o $@
clean:
	/bin/rm -f tau_upcr_wrap.o libtau_upcr_wrap.a

