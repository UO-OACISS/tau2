TAUROOT=
include $(TAUROOT)/include/Makefile

CC = $(TAU_CC)
CFLAGS = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_INTERNAL_FLAGS)
ARFLAGS = rcv

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: libTAU-preload$(TAU_SHLIBX)
	cp libTAU-preload$(TAU_SHLIBX) $(INSTALLDEST)/shared$(TAU_CONFIG)

taupreload.o: taupreload.c
	$(CC) $(CFLAGS) -c taupreload.c

libTAU-preload$(TAU_SHLIBX): taupreload.o
	$(CC) $(TAU_SHFLAGS) libTAU-preload$(TAU_SHLIBX) taupreload.o $(TAU_SHAREDLIBS) -ldl

clean:
	/bin/rm -f taupreload.o libTAU-preload$(TAU_SHLIBX)

