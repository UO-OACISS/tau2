include ${TAU_MAKEFILE} 
CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS)  $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -I.. $(TAU_SHMEM_INC)
EXTRA_FLAGS=

AR=$(TAU_AR)
ARFLAGS=rcv 

libtau_shmem_wrap.a: tau_shmem_wrap.o 
	$(AR) $(ARFLAGS) $@ $< 

tau_shmem_wrap.o: wr.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@
clean:
	/bin/rm -f tau_shmem_wrap.o libtau_shmem_wrap.a

