TAUROOT=/home/users/scottb/tau2
include $(TAUROOT)/include/Makefile

CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE) $(USER_OPT)

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib
TAU_OPENCL_SHARED=libTAU-OpenCL$(TAU_SHLIBX) #ENDIF##OPENCL#
#NOSHARED#TAU_OPENCL_SHARED= #ENDIF#

TAU_OPENCL_INSTALL_CMD=@echo "NOTE: Not building TAU's opencl DSO" 
TAU_OPENCL_INSTALL_CMD=cp $(TAU_OPENCL_SHARED) $(INSTALLDEST)/shared$(TAU_CONFIG) #ENDIF##OPENCL#

install:$(TAU_OPENCL_SHARED)
	$(TAU_OPENCL_INSTALL_CMD) 

$(TAU_OPENCL_SHARED): cl_wrap.o 
	$(TAU_CXX) $(TAU_SHFLAGS) $@ $< $(TAU_SHLIBS) -ldl

cl_wrap.o: cl_wrap.cpp $(TAU_INC_DIR)/Profile/TauGpuAdapterOpenCL.h
	$(TAU_CXX) -g $(CFLAGS) -I$(TAU_OPENCL_INC) -c $< -o $@
clean:
	/bin/rm -f cl_wrap.o libcl_wrap.so

