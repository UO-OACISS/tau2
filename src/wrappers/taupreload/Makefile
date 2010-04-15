TAUROOT=
include $(TAUROOT)/include/Makefile

CC = $(TAU_CC)
CFLAGS = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_INTERNAL_FLAGS)
ARFLAGS = rcv

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: libTAU-preload.so
	cp libTAU-preload.so $(INSTALLDEST)/shared$(TAU_CONFIG)

taupreload.o: taupreload.c
	$(CC) $(CFLAGS) -c taupreload.c

libTAU-preload.so: taupreload.o
	$(CC) $(TAU_SHFLAGS) libTAU-preload.so taupreload.o $(TAU_SHAREDLIBS) -ldl

clean:
	/bin/rm -f taupreload.o libTAU-preload.so

