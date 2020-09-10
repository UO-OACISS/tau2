include ../../../include/Makefile
include ../Makefile

CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE) $(USER_OPT) $(TAU_STDCXX11_FLAG)

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install:$(TAU_OPENCL_SHARED)
	$(TAU_SHARED_INSTALL_CMD) 

$(TAU_OPENCL_SHARED): cl_wrap.o 
	$(TAU_CXX) $(TAU_SHFLAGS) $@ $< -L$(TAU_LIB_DIR)/shared$(TAU_CONFIG) -lTAU $(TAU_LINKER_RPATH_OPT) -ldl

cl_wrap.o: cl_wrap.cpp $(TAU_INC_DIR)/Profile/TauGpuAdapterOpenCL.h
	$(TAU_CXX) -g $(CFLAGS) $(TAU_OPENCL_INCLUDE) $(BFDINCLUDE) -c $< -o $@
clean:
	/bin/rm -f cl_wrap.o $(TAU_OPENCL_SHARED) 

