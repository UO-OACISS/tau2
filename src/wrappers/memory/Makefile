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

TauMemoryWrap.o: TauMemoryWrap.cpp
	$(CXX) $(CFLAGS) -c TauMemoryWrap.cpp

libTAU-memorywrap$(TAU_SHLIBX): TauMemoryWrap.o
	$(CC) $(TAU_SHFLAGS) libTAU-memorywrap$(TAU_SHLIBX) TauMemoryWrap.o $(TAU_SHAREDLIBS) -ldl

clean:
	/bin/rm -f *.so *.o

