include ../../../include/Makefile
include ../Makefile

CC=$(TAU_CC) 
CFLAGS=$(PROFILEOPTS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE) $(TAU_STDCXX11_FLAG) -I. $(TAU_PERMISSIVE_FLAG)

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: $(TAU_CUDA_SHARED) $(TAU_CUDART_SHARED)
	$(TAU_SHARED_INSTALL_CMD) 

$(TAU_CUDART_SHARED): cudart_wrap.o 
	$(TAU_CUDA_CXX) -g $(TAU_SHFLAGS) $@ $< -L$(TAU_LIB_DIR)/shared$(TAU_CONFIG) -lTAU $(TAU_LINKER_RPATH_OPT) -ldl -L$(INSTALLDEST)/shared

cudart_wrap.o: cudart_wrap.cpp
	$(TAU_CUDA_CXX) -g $(TAU_INTERNAL_FLAGS) $(CFLAGS) -I$(TAU_CUDA_INC) -c $< -o $@

$(TAU_CUDA_SHARED): cuda_wrap.o
	$(TAU_CUDA_CXX) -g $(TAU_SHFLAGS) $@ $< -L$(TAU_LIB_DIR)/shared$(TAU_CONFIG) -lTAU $(TAU_LINKER_RPATH_OPT) -ldl -L$(INSTALLDEST)/shared

cuda_wrap.o: cuda_wrap.cpp
	$(TAU_CUDA_CXX) -g $(TAU_INTERNAL_FLAGS) $(CFLAGS) -I$(TAU_CUDA_INC) -c $< -o $@
clean:
	/bin/rm -f cudart_wrap.o cuda_wrap.o libTAU-CUDArt.so libTAU-CUDA.so

