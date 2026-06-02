include ../../../include/Makefile
include ../Makefile

CC=$(TAU_CC)
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE) $(TAU_CC_SPECIAL_OPTS) -fPIC

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: install-lbt install-server

install-lbt: libTAU-julia-blas.so
	mkdir -p $(INSTALLDEST)/wrappers/julia_blas_wrapper
	$(TAU_SHARED_INSTALL_CMD)

install-server: libTAU-julia-blas-server.so
	mkdir -p $(INSTALLDEST)/wrappers/julia_blas_wrapper
	$(TAU_SHARED_INSTALL_CMD)

libTAU-julia-blas.so: lbt_shim.o
	$(CC) $(TAU_SHFLAGS) $@ $< -ldl

libTAU-julia-blas-server.so: blas_server_shim.o
	$(CC) $(TAU_SHFLAGS) $@ $< -ldl -lpthread

lbt_shim.o: lbt_shim.c
	$(CC) $(CFLAGS) -c $< -o $@

blas_server_shim.o: blas_server_shim.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	/bin/rm -f lbt_shim.o blas_server_shim.o libTAU-julia-blas.so libTAU-julia-blas-server.so
