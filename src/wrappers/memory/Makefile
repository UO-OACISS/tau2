TAUROOT=
include $(TAUROOT)/include/Makefile

TAU_MEMORYWRAP_SHARED   = libTAU-memorywrap$(TAU_SHLIBX)

CC = $(TAU_CC)
CXX = $(TAU_CXX)
CFLAGS = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_INTERNAL_FLAGS) $(USER_OPT)
ARFLAGS = rcv

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: libTAU-memorywrap$(TAU_SHLIBX)
	cp libTAU-memorywrap$(TAU_SHLIBX) $(INSTALLDEST)/shared$(TAU_CONFIG)

memory_wrapper.o: memory_wrapper.c
	$(CC) $(CFLAGS) -c memory_wrapper.c

libTAU-memorywrap$(TAU_SHLIBX): memory_wrapper.o
	$(CC) $(TAU_SHFLAGS) libTAU-memorywrap$(TAU_SHLIBX) memory_wrapper.o $(TAU_SHAREDLIBS) -ldl

clean:
	/bin/rm -f *.so *.o

