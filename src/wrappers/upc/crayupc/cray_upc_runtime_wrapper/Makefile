include ${TAU_MAKEFILE} 
CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS) $(TAU_UPC_COMPILER_OPTIONS) $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -I..
EXTRA_FLAGS=

AR=$(TAU_AR)
ARFLAGS=rcv 

libcray_upc_runtime_wrap.a: cray_upc_runtime_wrap.o 
	$(AR) $(ARFLAGS) $@ $< 

cray_upc_runtime_wrap.o: wr.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@
clean:
	/bin/rm -f cray_upc_runtime_wrap.o libcray_upc_runtime_wrap.a

