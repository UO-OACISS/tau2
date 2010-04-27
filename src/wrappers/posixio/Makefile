TAUROOT=
include $(TAUROOT)/include/Makefile

TAU_IOWRAP_SHARED   = libTAU-iowrap.so

CC = $(TAU_CC)
CXX = $(TAU_CXX)
CFLAGS = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_INTERNAL_FLAGS)
ARFLAGS = rcv

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: libTauIOWrapper.a libTAU-iowrap.so
	cp libTauIOWrapper.a $(INSTALLDEST)
	cp libTAU-iowrap.so $(INSTALLDEST)/shared$(TAU_CONFIG)

libTauIOWrapper.a: iowrapper.o
	$(TAU_AR) $(ARFLAGS) libTauIOWrapper.a iowrapper.o

iowrapper.o: iowrapper.c
	$(CC) $(CFLAGS) -c iowrapper.c

iowrap_shared.o: iowrap_shared.cpp
	$(CXX) $(CFLAGS) -c iowrap_shared.cpp

libTAU-iowrap.so: iowrap_shared.o
	$(CC) $(TAU_SHFLAGS) libTAU-iowrap.so iowrap_shared.o $(TAU_SHAREDLIBS) -ldl

clean:
	/bin/rm -f iowrapper.o libTauIOWrapper.a iowrap_shared.o 

