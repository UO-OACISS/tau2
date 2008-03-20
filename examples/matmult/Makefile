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

F90=tau_f90.sh

LIBS= 
INCLUDE=
OBJS= matmult.o

matmult: $(OBJS)
	$(F90) $(OBJS) -o matmult $(LIBS)

.f90.o: 
	echo "Building $@ from $<"
	$(F90) -c $(INCLUDE) $< -o $@

clean:
	/bin/rm -f matmult.o matmult *.pdb profile.* *.inst.*
