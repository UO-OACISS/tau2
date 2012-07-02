TAUROOT=
include $(TAUROOT)/include/Makefile
CC=$(TAU_CC) 
ARFLAGS=rcv 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE) 
INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

LDPRELOAD_INSTALL_CMD=@echo "NOTE: Not building TAU's PTHREAD DSO" 
#PTHREAD_AVAILABLE#LDPRELOAD_INSTALL_CMD=cp $(PTHREAD_LDPRELOAD_LIB) $(INSTALLDEST)/shared$(TAU_CONFIG) #ENDIF#
#NOSHARED#TAU_CUDA_INSTALL_CMD=@echo "NOTE: Not building TAU's PTHREAD DSO" #ENDIF#

#PTHREAD_AVAILABLE#PTHREAD_LD_LIB=libTauPthreadWrap.a #ENDIF#
#PTHREAD_AVAILABLE#PTHREAD_LDPRELOAD_LIB=libTAU-pthread$(TAU_SHLIBX) #ENDIF#
#NOSHARED#PTHREAD_LDPRELOAD_LIB= #ENDIF#

install: install-ld-lib install-preload-lib

install-ld-lib: $(PTHREAD_LD_LIB)
	mkdir -p $(INSTALLDEST)/wrappers/pthread_wrapper
	cp link_options.tau $(INSTALLDEST)/wrappers/pthread_wrapper
	cp $(PTHREAD_LD_LIB) $(INSTALLDEST)

install-preload-lib: $(PTHREAD_LDPRELOAD_LIB)
	$(LDPRELOAD_INSTALL_CMD)

$(PTHREAD_LD_LIB): pthread_wrap_ld.o
	$(TAU_AR) $(ARFLAGS) $@ $<

pthread_wrap_ld.o: pthread_wrap.c
	$(CC) -DTAU_LIBRARY_SOURCE $(CFLAGS) -g -c $<  -o $@

$(PTHREAD_LDPRELOAD_LIB): pthread_wrap_preload.o
	$(CC) $(TAU_SHFLAGS) $(TAU_SHLIBS) -L$(INSTALLDEST) $< -o $@ -ldl

pthread_wrap_preload.o: pthread_wrap.c
	$(CC) -DTAU_LIBRARY_SOURCE -DTAU_PRELOAD_LIB $(CFLAGS) -g -c $< -o $@
