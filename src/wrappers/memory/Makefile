TAUROOT=
include $(TAUROOT)/include/Makefile

TAU_MEMORYWRAP_SHARED   = libTAU-memorywrap$(TAU_SHLIBX)
#NOSHARED#TAU_MEMORYWRAP_SHARED= #ENDIF#

TAU_MEMORYWRAP_INSTALL_CMD=cp $(TAU_MEMORYWRAP_SHARED) $(INSTALLDEST)/shared$(TAU_CONFIG)
#NOSHARED#TAU_MEMORYWRAP_INSTALL_CMD=@echo "NOTE: Not building TAU's memory wrapper DSO" #ENDIF#


CC = $(TAU_CC)
CXX = $(TAU_CXX)
CFLAGS = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_INTERNAL_FLAGS) $(USER_OPT)
ARFLAGS = rcv

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: $(TAU_MEMORYWRAP_SHARED)
	$(TAU_MEMORYWRAP_INSTALL_CMD)

memory_wrapper.o: memory_wrapper.c
	$(CC) $(CFLAGS) -c memory_wrapper.c

libTAU-memorywrap$(TAU_SHLIBX): memory_wrapper.o
	$(CC) $(TAU_SHFLAGS) libTAU-memorywrap$(TAU_SHLIBX) memory_wrapper.o $(TAU_SHAREDLIBS) -ldl

clean:
	/bin/rm -f *.so *.o

