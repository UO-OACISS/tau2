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
F90=TAU_MAKEFILE=$(TAU_MAKEFILE) tau_f90.sh

LIBS= 
INCLUDE=
OBJS= matmult.o
OMP_FLAG= -openmp

matmult: $(OBJS)
	$(F90) $(OBJS) -o matmult $(LIBS)

.f90.o: 
	echo "Building $@ from $<"
	$(F90) $(OMP_FLAG) -c $(INCLUDE) $< -o $@

clean:
	/bin/rm -f matmult.o matmult *.pdb profile.* *.inst.* *.continue.*

#intel compiler use -fpp for preprocessing + compiling
#mpiifort matmult.f90 -fpp -DUSE_OPENMP -o matmult_omp -openmp
