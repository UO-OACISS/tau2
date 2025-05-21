include ../../../include/Makefile
include ../Makefile


CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -fPIC 
EXTRA_FLAGS=

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: libTAU-gptl-wrap.so
	mkdir -p $(INSTALLDEST)/wrappers/gptl_wrapper
	$(TAU_SHARED_INSTALL_CMD)

libTAU-gptl-wrap.so: gptl_wrap_dynamic.o 
	$(CC) $(TAU_SHFLAGS) $@ $< $(TAU_SHLIBS) -ldl

gptl_wrap_dynamic.o: gptl_wrap_dynamic.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@

clean:
	/bin/rm -f gptl_wrap_dynamic.o libTAU-gptl-wrap.so

