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
F90=TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_f90.sh
FFLAGS=$(TAU_ALLOW_ARG_MISMATCH)

LIBS= 
INCLUDE=
OBJS= matmult.o

matmult: $(OBJS)
	$(F90) $(OBJS) -o matmult $(LIBS)

.f90.o: 
	echo "Building $@ from $<"
	$(F90) -c $(INCLUDE) $< -o $@ $(FFLAGS)

clean:
	/bin/rm -f matmult.o matmult *.pdb profile.* *.inst.* *.continue.*

