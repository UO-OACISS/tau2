###############################################################################
## Top level Makefile for Plugins #
###############################################################################

include ../../include/Makefile

PLUGININSTALLDIR = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/plugins

MAKE = make

CC = $(TAU_CC)

CXX = $(TAU_CXX)

CFLAGS = -fPIC -Wall -Werror

OBJS = alpha.o beta.o

SHAREDLIBS = alpha.so beta.so

all: install
	@echo

install: $(SHAREDLIBS)
	@mkdir -p $(PLUGININSTALLDIR)
	$(TAU_INSTALL) $^ $(PLUGININSTALLDIR)

alpha.so: alpha.o
	$(CC) -shared -o alpha.so alpha.o

alpha.o: alpha.c
	$(CC) $(CFLAGS) -c alpha.c

beta.so: beta.o
	$(CC) -shared -o beta.so beta.o

beta.o: beta.c
	$(CC) $(CFLAGS) -c beta.c
clean:
	/bin/rm -rf *.o $(SHAREDLIBS)

