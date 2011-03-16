TAUROOT=
include $(TAUROOT)/include/Makefile

TAU_ARMCI_SHARED=libTAU-armciwrap$(TAU_SHLIBX)
#CRAYXMT#TAU_ARMCI_SHARED= #ENDIF#
#BGL#TAU_ARMCI_SHARED= #ENDIF#
#CATAMOUNT#TAU_ARMCI_SHARED= #ENDIF#
#CRAYX1CC#TAU_ARMCI_SHARED= #ENDIF#
#SUNCC#TAU_ARMCI_SHARED= #ENDIF#
#APPLEPGI#TAU_ARMCI_SHARED= #ENDIF#
#HITACHI#TAU_ARMCI_SHARED= #ENDIF#
#CRAYXMT#TAU_ARMCI_SHARED= #ENDIF#
#CRAYCC#TAU_ARMCI_SHARED= #ENDIF#
#OPEN64ORC#TAU_ARMCI_SHARED= #ENDIF#
#USE_NECCXX#TAU_ARMCI_SHARED= #ENDIF#
#NOSHARED#TAU_ARMCI_SHARED= #ENDIF#

TAU_ARMCI_INSTALL_CMD=cp $(TAU_ARMCI_SHARED) $(INSTALLDEST)/shared$(TAU_CONFIG)
#NOSHARED#TAU_ARMCI_INSTALL_CMD=@echo "NOTE: Not building TAU's armciwrap DSO"  #ENDIF#

CC = $(TAU_CC)
CXX = $(TAU_CXX)
CFLAGS = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_INTERNAL_FLAGS) $(USER_OPT) -I$(ARMCIDIR)/include
ARFLAGS = rcv

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: libTauARMCIWrapper.a $(TAU_ARMCI_SHARED)
	cp libTauARMCIWrapper.a $(INSTALLDEST)
	$(TAU_ARMCI_INSTALL_CMD)

libTauARMCIWrapper.a: armciwrapper.o
	$(TAU_AR) $(ARFLAGS) libTauARMCIWrapper.a armciwrapper.o

armciwrapper.o: armciwrapper.c
	$(CC) $(CFLAGS) -c armciwrapper.c

armciwrap_shared.o: armciwrapper.c
	$(CC) $(CFLAGS) $(TAU_INTERNAL_FLAGS) -c armciwrapper.c -o armciwrap_shared.o

libTAU-armciwrap$(TAU_SHLIBX): armciwrap_shared.o
	$(CC) $(TAU_SHFLAGS) libTAU-armciwrap$(TAU_SHLIBX) armciwrap_shared.o $(TAU_SHAREDLIBS) -ldl -L$(ARMCIDIR)/lib -larmci -lga

clean:
	/bin/rm -f armciwrapper.o libTauARMCIWrapper.a armciwrap_shared.o libTAU-armciwrap$(TAU_SHLIBX)

