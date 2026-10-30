TAU_MAKEFILE ?=../../../../include/Makefile

include $(TAU_MAKEFILE)

F90 = SALT_BUILD_USES_SALT_FLANG=1 TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_f90.sh -optSaltInst -optNoRevert

FFLAGS = -g -O2

all: matmult

matmult: matmult.o
	$(F90) $(FFLAGS) matmult.o -o $@

matmult.o: matmult.f90
	$(F90) $(FFLAGS) -c $< -o $@

clean:
	/bin/rm -f matmult matmult.o *.mod *.saltmod profile.* *.inst.*
