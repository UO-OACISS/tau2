TAUROOT=
include $(TAUROOT)/include/Makefile

CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE)  -I.. 

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib
TAU_OPENCL_SHARED=libTAU-OpenCL$(TAU_SHLIBX)
#NOSHARED#TAU_PRELOAD_SHARED= #ENDIF#

TAU_OPENCL_INSTALL_CMD=cp $(TAU_OPENCL_SHARED) $(INSTALLDEST)/shared$(TAU_CONFIG)
#NOSHARED#TAU_PRELOAD_INSTALL_CMD=@echo "NOTE: Not building TAU's opencl DSO" #ENDIF#

install:$(TAU_OPENCL_SHARED)
	$(TAU_OPENCL_INSTALL_CMD) 

libTAU-OpenCL$(TAU_SHLIBX): cl_wrap.o 
	$(CC) $(TAU_SHFLAGS) $@ $< $(TAU_SHLIBS) -ldl

cl_wrap.o: cl_wrap.cpp
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	/bin/rm -f cl_wrap.o libcl_wrap.so

