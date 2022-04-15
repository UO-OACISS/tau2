include ../../../include/Makefile
include ../Makefile


CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) $(CRAY_DMAPP_INCLUDE_OPTS) $(CRAY_GNI_HEADERS_INCLUDE_OPTS) -fPIC
EXTRA_FLAGS=

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: libTAU-gni-wrap.so
	mkdir -p $(INSTALLDEST)/wrappers/gni_wrapper
	$(TAU_SHARED_INSTALL_CMD)

libTAU-gni-wrap.so: gni_wrap_dynamic.o 
	$(CC) $(TAU_SHFLAGS) $@ $< $(TAU_SHLIBS) -ldl

gni_wrap_dynamic.o: gni_wrap_dynamic.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@

clean:
	/bin/rm -f gni_wrap_dynamic.o libTAU-gni_wrap_dynamic.so

