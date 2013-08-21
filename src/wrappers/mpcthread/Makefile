include ../../../include/Makefile
include ../Makefile
CC=$(TAU_CC) 
ARFLAGS=rcv 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE) 
INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: install-ld-lib install-preload-lib

install-ld-lib: $(PTHREAD_LD_LIB)
	mkdir -p $(INSTALLDEST)/wrappers/mpcthread_wrapper
	$(TAU_STATIC_INSTALL_CMD)

install-preload-lib: $(PTHREAD_LDPRELOAD_LIB)
	$(TAU_SHARED_INSTALL_CMD)

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
