include ../../../include/Makefile
include ../Makefile
CC=$(TAU_CC)
ARFLAGS=rcv
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE) $(TAU_CC_SPECIAL_OPTS)
INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: $(INSTALL_LD_LIB) $(INSTALL_PRELOAD_LIB)

$(INSTALL_LD_LIB): $(PTHREAD_LDPRELOAD_LIB)
	mkdir -p $(INSTALLDEST)/wrappers/pthread_wrapper
	cp link_options.tau $(INSTALLDEST)/wrappers/pthread_wrapper
	$(TAU_SHARED_INSTALL_CMD)

install-preload-lib: $(PTHREAD_LD_LIB)
	$(TAU_STATIC_INSTALL_CMD)

$(PTHREAD_LD_LIB): pthread_wrap_ld.o
	$(TAU_AR) $(ARFLAGS) $@ $<

pthread_wrap_ld.o: pthread_wrap.c
	$(CC) -DTAU_LIBRARY_SOURCE $(CFLAGS) -g -c $<  -o $@

$(PTHREAD_LDPRELOAD_LIB): pthread_wrap_preload.o
	$(CC) $(TAU_SHFLAGS) $@ $< -ldl

pthread_wrap_preload.o: pthread_wrap.c
	$(CC) -DTAU_LIBRARY_SOURCE -DTAU_PRELOAD_LIB $(CFLAGS) -g -c $< -o $@

# *CWL* This is REQUIRED for any component of TAU for a successful new build!
clean:
	/bin/rm -f pthread_wrap_preload.o pthread_wrap_ld.o $(PTHREAD_LD_LIB) $(PTHREAD_LDPRELOAD_LIB)
