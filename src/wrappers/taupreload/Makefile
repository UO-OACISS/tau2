include ../../../include/Makefile
include ../Makefile

CFLAGS = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_INTERNAL_FLAGS) $(TAU_CC_SPECIAL_OPTS)
ARFLAGS = rcv

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install:$(TAU_PRELOAD_SHARED) $(TAU_AUDITOR_SHARED)
	$(TAU_SHARED_INSTALL_CMD)

taupreload.o: taupreload.c
	$(TAU_CC) $(CFLAGS) -c taupreload.c

libTAU-preload$(TAU_SHLIBX): taupreload.o
	$(TAU_CC) $(TAU_SHFLAGS) libTAU-preload$(TAU_SHLIBX) taupreload.o $(TAU_LIBDL)

dl_auditor.o: dl_auditor.c
	$(TAU_CC) $(CFLAGS) -c dl_auditor.c

libTAU-dl-auditor$(TAU_SHLIBX): dl_auditor.o
	$(TAU_CC) -lc -ldl $(TAU_SHFLAGS) libTAU-dl-auditor$(TAU_SHLIBX) dl_auditor.o -ldl

clean:
	/bin/rm -f taupreload.o libTAU-preload$(TAU_SHLIBX) dl_auditor.o libTAU-dl-auditor$(TAU_SHLIBX)

