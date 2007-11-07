TAUROOT=
include $(TAUROOT)/include/Makefile

CC = $(TAU_CC)
CFLAGS = $(TAU_INCLUDE) $(TAU_DEFS)
ARFLAGS = rcv

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: libTauIOWrapper.a
	cp libTauIOWrapper.a $(INSTALLDEST)

libTauIOWrapper.a: iowrapper.o
	$(TAU_AR) $(ARFLAGS) libTauIOWrapper.a iowrapper.o

iowrapper.o: iowrapper.c
	$(CC) $(CFLAGS) -c iowrapper.c

clean:
	/bin/rm -f iowrapper.o libTauIOWrapper.a

