TAUROOT=
include $(TAUROOT)/include/Makefile

CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE)

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib
TAU_OPENCL_SHARED=libTAU-OpenCL$(TAU_SHLIBX)
#NOSHARED#TAU_OPENCL_SHARED= #ENDIF#

TAU_OPENCL_INSTALL_CMD=cp $(TAU_OPENCL_SHARED) $(INSTALLDEST)/shared$(TAU_CONFIG)
#NOSHARED#TAU_OPENCL_INSTALL_CMD=@echo "NOTE: Not building TAU's opencl DSO" #ENDIF#

ifneq ($(TAU_OPENCL_INC), )
install:$(TAU_OPENCL_SHARED)
	$(TAU_OPENCL_INSTALL_CMD) 
else
install:
endif

$(TAU_OPENCL_SHARED): cl_wrap.o 
	$(TAU_CXX) $(TAU_SHFLAGS) $@ $< $(TAU_SHLIBS) -ldl

cl_wrap.o: cl_wrap.cpp $(TAU_INC_DIR)/Profile/TauGpuAdapterOpenCL.h
	$(TAU_CXX) -g $(CFLAGS) -I$(TAU_OPENCL_INC) -c $< -o $@
clean:
	/bin/rm -f cl_wrap.o libcl_wrap.so

