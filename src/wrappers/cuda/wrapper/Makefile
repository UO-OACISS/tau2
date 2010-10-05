include ${TAU_MAKEFILE} 
CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE)  -I.. 

libcuda_runtime_api_wrap.so: cuda_runtime_api_wrap.o 
	$(CC) $(TAU_SHFLAGS) $@ $< $(TAU_SHLIBS) -ldl

cuda_runtime_api_wrap.o: cuda_runtime_api.h.ins
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	/bin/rm -f cuda_runtime_api_wrap.o libcuda_runtime_api_wrap.so

