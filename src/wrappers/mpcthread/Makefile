TAUROOT=
include $(TAUROOT)/include/Makefile
CC=$(TAU_CC) 
ARFLAGS=rcv 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE) 
INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib
TAU_STATIC_INSTALL_CMD=@echo "NOTE: Not building MPC Thread Library"

LDPRELOAD_INSTALL_CMD=@echo "NOTE: Not building TAU's PTHREAD DSO" 
#PTHREAD_AVAILABLE#LDPRELOAD_INSTALL_CMD=cp $(PTHREAD_LDPRELOAD_LIB) $(INSTALLDEST)/shared$(TAU_CONFIG) #ENDIF#
#NOSHARED#LDPRELOAD_INSTALL_CMD= #ENDIF#
#NOSHARED#TAU_CUDA_INSTALL_CMD=@echo "NOTE: Not building TAU's PTHREAD DSO" #ENDIF#

#PTHREAD_AVAILABLE#PTHREAD_LD_LIB=libTauMPCThreadWrap.a #ENDIF#
#PTHREAD_AVAILABLE#PTHREAD_LDPRELOAD_LIB=libTAU-MPCThread$(TAU_SHLIBX) #ENDIF#
#NOSHARED#PTHREAD_LDPRELOAD_LIB= #ENDIF#

install: install-ld-lib install-preload-lib

install-ld-lib: $(PTHREAD_LD_LIB)
	mkdir -p $(INSTALLDEST)/wrappers/mpcthread_wrapper
	$(TAU_STATIC_INSTALL_CMD)

install-preload-lib: $(PTHREAD_LDPRELOAD_LIB)
	$(LDPRELOAD_INSTALL_CMD)

$(PTHREAD_LD_LIB): mpcthread_wrap_ld.o
	$(TAU_AR) $(ARFLAGS) $@ $<

mpcthread_wrap_ld.o: mpcthread_wrap.c
	$(CC) -DTAU_LIBRARY_SOURCE $(CFLAGS) -g -c $<  -o $@

$(PTHREAD_LDPRELOAD_LIB): mpcthread_wrap_preload.o
	$(CC) $(TAU_SHFLAGS) $@ $(TAU_SHLIBS) -L$(INSTALLDEST) $< -ldl

mpcthread_wrap_preload.o: mpcthread_wrap.c
	$(CC) -DTAU_LIBRARY_SOURCE -DTAU_PRELOAD_LIB $(CFLAGS) -g -c $< -o $@

clean:
	/bin/rm -f *.o *.a *.so
