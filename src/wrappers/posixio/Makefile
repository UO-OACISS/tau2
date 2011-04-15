TAUROOT=
include $(TAUROOT)/include/Makefile

TAU_IOWRAP_SHARED=libTAU-iowrap$(TAU_SHLIBX)
#CRAYXMT#TAU_IOWRAP_SHARED= #ENDIF#
#BGL#TAU_IOWRAP_SHARED= #ENDIF#
#CATAMOUNT#TAU_IOWRAP_SHARED= #ENDIF#
#CRAYX1CC#TAU_IOWRAP_SHARED= #ENDIF#
#SUNCC#TAU_IOWRAP_SHARED= #ENDIF#
#APPLEPGI#TAU_IOWRAP_SHARED= #ENDIF#
#HITACHI#TAU_IOWRAP_SHARED= #ENDIF#
#CRAYXMT#TAU_IOWRAP_SHARED= #ENDIF#
#CRAYCC#TAU_IOWRAP_SHARED= #ENDIF#
#OPEN64ORC#TAU_IOWRAP_SHARED= #ENDIF#
#USE_NECCXX#TAU_IOWRAP_SHARED= #ENDIF#
#NOSHARED#TAU_IOWRAP_SHARED= #ENDIF#

TAU_IOWRAP_INSTALL_CMD=cp $(TAU_IOWRAP_SHARED) $(INSTALLDEST)/shared$(TAU_CONFIG)
#NOSHARED#TAU_IOWRAP_INSTALL_CMD=@echo "NOTE: Not building TAU's iowrap DSO"  #ENDIF#

CC = $(TAU_CC)
CXX = $(TAU_CXX)
#PGI#PGIFLAG=-c89#ENDIF#
CFLAGS = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_INTERNAL_FLAGS) $(USER_OPT) $(PGIFLAG) 
ARFLAGS = rcv

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install: libTauIOWrapper.a libTauPosixWrap.a $(TAU_IOWRAP_SHARED)
	cp libTauIOWrapper.a libTauPosixWrap.a $(INSTALLDEST)
	mkdir -p $(INSTALLDEST)/wrappers/io_wrapper
	cp link_options.tau $(INSTALLDEST)/wrappers/io_wrapper
	$(TAU_IOWRAP_INSTALL_CMD)

libTauPosixWrap.a: iowrap_link.o
	$(TAU_AR) $(ARFLAGS) libTauPosixWrap.a iowrap_link.o

libTauIOWrapper.a: iowrapper.o
	$(TAU_AR) $(ARFLAGS) libTauIOWrapper.a iowrapper.o

iowrapper.o: iowrapper.c
	$(CC) $(CFLAGS) -c iowrapper.c

iowrap_link.o: iowrap_link.c
	$(CC) $(CFLAGS) -c iowrap_link.c

iowrap_shared.o: iowrap_shared.c
	$(CC) $(CFLAGS) -c iowrap_shared.c

libTAU-iowrap$(TAU_SHLIBX): iowrap_shared.o
	$(CC) $(TAU_SHFLAGS) libTAU-iowrap$(TAU_SHLIBX) iowrap_shared.o $(TAU_SHAREDLIBS) -ldl

clean:
	/bin/rm -f iowrapper.o libTauIOWrapper.a iowrap_shared.o iowrap_link.o libTauPosixWrap.a

