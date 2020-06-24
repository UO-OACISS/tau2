#****************************************************************************
#*					  TAU Performance System (R)						  *
#*					  http://tau.uoregon.edu							  *
#****************************************************************************
#*	Copyright 2008													   **
#*	Department of Computer and Information Science, University of Oregon **
#*	Juelich Supercomputing Center, NIC, Germany						  **
#*	Advanced Computing Laboratory, Los Alamos National Laboratory		**
#****************************************************************************
# Use gmake to build matmult

.SUFFIXES : .f90

TAU_MAKEFILE ?=../../../include/Makefile
include $(TAU_MAKEFILE)
F90=TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_f90.sh

LIBS=
INCLUDE=
OBJS= matmult.o
.PHONY: all

all: clean matmult test

matmult: $(OBJS)
	$(F90) $(OBJS) -o matmult $(LIBS)

.f90.o:
	echo "Building $@ from $<"
	$(F90) -c $(INCLUDE) $< -o $@

clean:
	/bin/rm -f matmult.o matmult *.pdb profile.* *.inst.* *.continue.*

test:
	$(MPIRUN) ./matmult
	test -f profile.0.0.0 || test -d MULTI__PAPI_TOT_INS || exit 1
	$(PPROF_CMD)


