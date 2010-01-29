include ../common.mk
include ${TAU_MAKEFILE} 
CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE) -I$(CUDA_BASE)/include

libcuda_runtime_api_wrap.so: cuda_runtime_api_wrap.o 
	$(CC) -shared -o ../release/$@ $< $(TAU_SHLIBS) -ldl

cuda_runtime_api_wrap.o: cudart_wrapper.c
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	/bin/rm -f cuda_runtime_api_wrap.o ../release/libcuda_runtime_api_wrap.so
