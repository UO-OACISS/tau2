include ${TAU_MAKEFILE} 
CC=$(TAU_CC) 
ARFLAGS=rcv 
CFLAGS=$(TAU_DEFS) $(TAU_UPC_COMPILER_OPTIONS) $(UPCC_C_PREFIX)$(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -I.. 

libcray_upc_runtime_wrap.a: cray_upc_runtime_wrap.o 
	$(TAU_AR) $(ARFLAGS) $@ $< 

cray_upc_runtime_wrap.o: wr.c
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	/bin/rm -f cray_upc_runtime_wrap.o libcray_upc_runtime_wrap.a

