include ../../../include/Makefile
include ../Makefile
CC=$(TAU_CC) 
ARFLAGS=rcv 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE) $(TAU_CC_SPECIAL_OPTS) 
INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: $(INSTALL_LD_LIB) $(INSTALL_PRELOAD_LIB)

$(INSTALL_LD_LIB): $(GOMP_LD_LIB)
	mkdir -p $(INSTALLDEST)/wrappers/gomp_wrapper
	cp link_options.tau $(INSTALLDEST)/wrappers/gomp_wrapper
	$(TAU_STATIC_INSTALL_CMD)

install-preload-lib: $(GOMP_LDPRELOAD_LIB)
	$(TAU_SHARED_INSTALL_CMD)

$(GOMP_LD_LIB): gomp_wrap.o omp_collector_util.o gomp_timers.o
	$(TAU_AR) $(ARFLAGS) $@ $< omp_collector_util.o gomp_timers.o

.c.o: 
	$(CC) -DTAU_LIBRARY_SOURCE $(CFLAGS) -g -c $<  -o $@

$(GOMP_LDPRELOAD_LIB): gomp_wrap_preload.o omp_collector_util.o gomp_timers.o
	# $(CC) $(TAU_SHFLAGS) $@ $(TAU_SHLIBS) -L$(INSTALLDEST) $< omp_collector_util.o gomp_timers.o -ldl -Wl,--version-script=exports_so.txt
	$(CC) $(TAU_SHFLAGS) $@ $(TAU_SHLIBS) -L$(INSTALLDEST) $< omp_collector_util.o gomp_timers.o -ldl

gomp_wrap_preload.o: gomp_wrap.c
	$(CC) -DTAU_LIBRARY_SOURCE -DTAU_PRELOAD_LIB $(CFLAGS) -g -c $< -o $@

# *CWL* This is REQUIRED for any component of TAU for a successful new build!
clean:
	/bin/rm -f *.o $(GOMP_LD_LIB) $(GOMP_LDPRELOAD_LIB)
