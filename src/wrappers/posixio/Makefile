include ../../../include/Makefile
include ../Makefile

CC = $(TAU_CC)
CXX = $(TAU_CXX)
CFLAGS = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_INTERNAL_FLAGS) $(USER_OPT) $(PGIFLAG) 
ARFLAGS = rcv

INSTALLDEST = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

install-static-io: libTauIOWrapper.a
	$(TAU_STATIC_INSTALL_CMD)

install-static-posix: libTauPosixWrap.a
	$(TAU_STATIC_INSTALL_CMD)

install-shared: $(TAU_IOWRAP_SHARED)
	$(TAU_SHARED_INSTALL_CMD)

install: install-static-io install-static-posix install-shared
	mkdir -p $(INSTALLDEST)/wrappers/io_wrapper
	sed 's#UNWIND_LIB_FLAG#$(UNWIND_LIB_FLAG)#' link_options.tau.master > $(INSTALLDEST)/wrappers/io_wrapper/link_options.tau

libTauPosixWrap.a: iowrap_link.o
	$(TAU_AR) $(ARFLAGS) libTauPosixWrap.a iowrap_link.o

libTauIOWrapper.a: iowrapper.o
	$(TAU_AR) $(ARFLAGS) libTauIOWrapper.a iowrapper.o

iowrapper.o: iowrapper.c
	$(CC) $(CFLAGS) -c iowrapper.c -I.

iowrap_link.o: iowrap_link.c
	$(CC) $(CFLAGS) -c iowrap_link.c -I.

iowrap_shared.o: iowrap_shared.c
	$(CC) $(CFLAGS) -c iowrap_shared.c -I.

libTAU-iowrap$(TAU_SHLIBX): iowrap_shared.o
	$(CC) $(TAU_SHFLAGS) libTAU-iowrap$(TAU_SHLIBX) iowrap_shared.o $(TAU_SHAREDLIBS) -ldl

clean:
	/bin/rm -f iowrapper.o libTauIOWrapper.a iowrap_shared.o iowrap_link.o libTauPosixWrap.a

