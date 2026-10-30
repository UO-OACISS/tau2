TAU_MAKEFILE ?=../../../../include/Makefile

include $(TAU_MAKEFILE)

CC = TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_cc.sh -optSaltInst -optNoRevert

CFLAGS = -g -O2 $(TAU_OPENMP_OPTION)

all: matmult

matmult: matmult.o
	$(CC) $(CFLAGS) matmult.o -o $@

matmult.o: matmult.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	/bin/rm -f matmult matmult.o profile.* *.inst.*
