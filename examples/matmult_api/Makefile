#****************************************************************************
#*                      TAU Performance System (R)                          *
#*                      http://tau.uoregon.edu                              *
#****************************************************************************
#*    Copyright 2008                                                       **
#*    Department of Computer and Information Science, University of Oregon **
#*    Juelich Supercomputing Center, NIC, Germany                          **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
# Use gmake to build matmult

.SUFFIXES : .f90

TAU_MAKEFILE ?=../../include/Makefile
include $(TAU_MAKEFILE)
F90=TAU_MAKEFILE=$(TAU_MAKEFILE) tau_f90.sh -optLinkOnly

LIBS=
INCLUDE=
OBJS= matmult.ss.o

matmult.ss: $(OBJS)
	$(F90) $(OBJS) -o matmult.ss $(LIBS)

.f90.o:
	echo "Building $@ from $<"
	$(F90) -c $(INCLUDE) $< -o $@

clean:
	/bin/rm -f matmult.ss.o matmult.ss profile.*
