TAU_MAKEFILE ?=../../../../include/Makefile

include $(TAU_MAKEFILE)

F90 = TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_f90.sh -optSaltInst -optTauSelectFile=select.tau

FFLAGS = -g -O2

OBJS = matmult.o matmult_impl.o

all: matmult

matmult: $(OBJS)
	$(F90) $(FFLAGS) $(OBJS) -o $@

matmult_impl.o: matmult.o

%.o: %.f90
	$(F90) $(FFLAGS) -c $< -o $@

clean:
	/bin/rm -f matmult $(OBJS) *.mod *.smod *.saltmod profile.* *.inst.*
