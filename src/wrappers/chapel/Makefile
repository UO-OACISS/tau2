include ../../../include/Makefile
include ../Makefile


CC=$(TAU_CC) 
CFLAGS=$(TAU_DEFS) $(TAU_INTERNAL_FLAG1) $(TAU_INCLUDE)  $(TAU_MPI_INCLUDE) -fPIC 
EXTRA_FLAGS=

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: libTAU-chapel-wrap.so
	mkdir -p $(INSTALLDEST)/wrappers/chapel_wrapper
	$(TAU_SHARED_INSTALL_CMD)

libTAU-chapel-wrap.so: chapel_wrap_dynamic.o 
	$(CC) $(TAU_SHFLAGS) $@ $< $(TAU_SHLIBS) -ldl

chapel_wrap_dynamic.o: chapel_wrap_dynamic.c
	$(CC) $(CFLAGS) $(EXTRA_FLAGS) -c $< -o $@

clean:
	/bin/rm -f chapel_wrap_dynamic.o libTAU-chapel-wrap.so

