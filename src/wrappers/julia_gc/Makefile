include ../../../include/Makefile
include ../Makefile

CC=$(TAU_CC)
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE) $(TAU_CC_SPECIAL_OPTS) -fPIC

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: libTAU-julia-gc.so
	mkdir -p $(INSTALLDEST)/wrappers/julia_gc_wrapper
	$(TAU_SHARED_INSTALL_CMD)

libTAU-julia-gc.so: julia_gc_shim.o
	$(CC) $(TAU_SHFLAGS) $@ $< -ldl

julia_gc_shim.o: julia_gc_shim.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	/bin/rm -f julia_gc_shim.o libTAU-julia-gc.so
