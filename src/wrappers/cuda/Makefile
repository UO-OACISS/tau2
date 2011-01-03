TAUROOT=
include $(TAUROOT)/include/Makefile

CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE) -I.

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib
TAU_CUDA_SHARED=libTAU-CUDA$(TAU_SHLIBX)
TAU_CUDART_SHARED=libTAU-CUDArt$(TAU_SHLIBX)
ifneq ($(TAU_CUDA_INC), )
install:$(TAU_CUDA_SHARED) $(TAU_CUDART_SHARED)
	$(TAU_CUDA_INSTALL_CMD) 
else
install:
endif
#NOSHARED#TAU_CUDA_SHARED= #ENDIF#

TAU_CUDA_INSTALL_CMD=cp $(TAU_CUDA_SHARED) $(TAU_CUDART_SHARED) $(INSTALLDEST)/shared$(TAU_CONFIG)
#NOSHARED#TAU_CUDA_INSTALL_CMD=@echo "NOTE: Not building TAU's CUDA DSO" #ENDIF#


$(TAU_CUDART_SHARED): cudart_wrap.o 
	$(TAU_CXX) -g $(TAU_SHFLAGS) $@ $< $(TAU_SHLIBS) -ldl -L$(INSTALLDEST)/shared

cudart_wrap.o: cudart_wrap.cpp
	$(TAU_CXX) $(CFLAGS) -I$(TAU_CUDA_INC) -c $< -o $@

$(TAU_CUDA_SHARED): cuda_wrap.o
	$(TAU_CXX) -g $(TAU_SHFLAGS) $@ $< $(TAU_SHLIBS) -ldl -L$(INSTALLDEST)/shared

cuda_wrap.o: cuda_wrap.c
	$(TAU_CXX) $(CFLAGS) -I$(TAU_CUDA_INC) -c $< -o $@
clean:
	/bin/rm -f cuda_wrap.o libTAU_CUDA.so

