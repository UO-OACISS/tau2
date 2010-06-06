TAUROOT=
include $(TAUROOT)/include/Makefile

CC = $(TAU_CC)
CFLAGS = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_INTERNAL_FLAGS)
ARFLAGS = rcv

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib
TAU_PRELOAD_SHARED=libTAU-preload$(TAU_SHLIBX)
#NOSHARED#TAU_PRELOAD_SHARED= #ENDIF#

TAU_PRELOAD_INSTALL_CMD=cp $(TAU_PRELOAD_SHARED) $(INSTALLDEST)/shared$(TAU_CONFIG)
#NOSHARED#TAU_PRELOAD_INSTALL_CMD=@echo "NOTE: Not building TAU's preload DSO" #ENDIF#

install:$(TAU_PRELOAD_SHARED)
	$(TAU_PRELOAD_INSTALL_CMD) 

taupreload.o: taupreload.c
	$(CC) $(CFLAGS) -c taupreload.c

libTAU-preload$(TAU_SHLIBX): taupreload.o
	$(CC) $(TAU_SHFLAGS) libTAU-preload$(TAU_SHLIBX) taupreload.o $(TAU_SHAREDLIBS) -ldl

clean:
	/bin/rm -f taupreload.o libTAU-preload$(TAU_SHLIBX)

