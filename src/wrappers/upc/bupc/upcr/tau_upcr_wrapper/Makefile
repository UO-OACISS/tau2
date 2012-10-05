include ${TAU_MAKEFILE} 
CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS)  $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -I..
EXTRA_FLAGS=

AR=$(TAU_AR)
ARFLAGS=rcv 

libtau_upcr_wrap.a: tau_upcr_wrap.o 
	$(AR) $(ARFLAGS) $@ $< 

tau_upcr_wrap.o: wr.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@
clean:
	/bin/rm -f tau_upcr_wrap.o libtau_upcr_wrap.a

