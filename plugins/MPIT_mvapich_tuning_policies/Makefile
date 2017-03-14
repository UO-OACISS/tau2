###############################################################################
## Top level Makefile for Plugins #
###############################################################################

include ../../include/Makefile

PLUGININSTALLDIR = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/plugins

MAKE = make

CC = $(TAU_CC)

CXX = $(TAU_CXX)

CFLAGS = $(TAU_MPI_COMPILE_INCLUDE) $(TAU_INCLUDE) $(TAU_DEFS) $(USER_OPT) $(TAU_INTERNAL_FLAGS) -fPIC

LDFLAGS = $(TAU_MPI_LIB)

OBJS = libfree_unused_vbufs.so free_unused_vbufs.o

SHAREDLIBS = libfree_unused_vbufs.so

all: install
	@echo

install: $(SHAREDLIBS)
	@mkdir -p $(PLUGININSTALLDIR)
	$(TAU_INSTALL) $^ $(PLUGININSTALLDIR)

libfree_unused_vbufs.so: free_unused_vbufs.o
	$(CC) $(LDFLAGS) -shared -o libfree_unused_vbufs.so free_unused_vbufs.o

free_unused_vbufs.o:      free_unused_vbufs.c
	$(CC) $(CFLAGS)  -c free_unused_vbufs.c

clean:
	/bin/rm -rf *.o $(SHAREDLIBS)
