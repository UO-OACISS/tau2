#****************************************************************************
#*                      TAU Portable Profiling Package                     **
#*                      http://www.cs.uoregon.edu/research/tau             **
#****************************************************************************
#*    Copyright 2004                                                       **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
# Use gmake to build ring
TAUROOTDIR      = ../../..

.SUFFIXES : .f90

include $(TAUROOTDIR)/include/Makefile

# We use TAU_COMPILER (defined in the above stub Makefile) for instrumenting
# the source code. We can pass it optional arguments. See tau_compiler.sh -help

F90=$(TAUROOTDIR)/$(TAU_ARCH)/bin/tau_f90.sh
LIBS= 
INCLUDE=
OBJS= ring.o

ring: $(OBJS)
	$(F90) $(OBJS) -o ring $(LIBS)

.f90.o: 
	echo "Building $@ from $<"
	$(F90) -c $(INCLUDE) $< -o $@

clean:
	/bin/rm -f ring.o ring *.pdb profile.* 
