#****************************************************************************
#*                      TAU Portable Profiling Package                     **
#*                      http://www.cs.uoregon.edu/research/tau             **
#****************************************************************************
#*    Copyright 2004                                                       **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
# Use gmake to build ring

.SUFFIXES : .f90

TAU_MAKEFILE ?=../../../include/Makefile
include $(TAU_MAKEFILE)
F90=TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_f90.sh

LIBS=
INCLUDE=
OBJS= ring.o
.PHONY: all

all: clean ring test

ring: $(OBJS)
	$(F90) $(OBJS) -o ring $(LIBS)

.f90.o:
	echo "Building $@ from $<"
	$(F90) -c $(INCLUDE) $< -o $@

clean:
	/bin/rm -f ring.o ring *.pdb profile.*

test:
	$(MPIRUN) ./ring
	test -f profile.0.0.0 || test -d MULTI__PAPI_TOT_INS || exit 1
	$(PPROF_CMD)


