include ${TAU_MAKEFILE}
CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS)  $(TAU_INCLUDE) $(TAU_MPI_INCLUDE)  -I.. $(TAU_SHMEM_INC) -fPIC
EXTRA_FLAGS=$(TAU_CRAY_SHMEM_EXTRA_DEFS)

AR=$(TAU_AR)
ARFLAGS=rcv 

all: libtau_shmem_wrap.a libtau_shmem_wrap.so 
libtau_shmem_wrap.so: tau_shmem_wrap_static.o tau_shmem_wrap_dynamic.o 
	$(CC) $(TAU_SHFLAGS) $@ $^ $(TAU_SHLIBS) -ldl

libtau_shmem_wrap.a: tau_shmem_wrap_static.o 
	$(AR) $(ARFLAGS) $@ $^ 

tau_shmem_wrap_dynamic.o: wr_dynamic.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@
tau_shmem_wrap_static.o: wr.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@

clean:
	/bin/rm -f tau_shmem_wrap_dynamic.o tau_shmem_wrap_static.o libtau_shmem_wrap.so libtau_shmem_wrap.a

