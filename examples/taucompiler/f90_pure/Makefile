#****************************************************************************
#*                      TAU Portable Profiling Package                     **
#*                      http://www.cs.uoregon.edu/research/tau             **
#****************************************************************************
#*    Copyright 2004                                                       **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
# Use gmake to build pure
TAUROOTDIR      = ../../..

.SUFFIXES : .f90

include $(TAUROOTDIR)/include/Makefile

# We use TAU_COMPILER (defined in the above stub Makefile) for instrumenting
# the source code. We can pass it optional arguments. See tau_compiler.sh -help

OPTS = -optVerbose -optKeepFiles  -optCompile=$(TAU_F90_SUFFIX)
F90 = $(TAU_COMPILER) $(OPTS) $(TAU_F90) 
LIBS= 
INCLUDE=
OBJS= pure.o

pure: $(OBJS)
	$(F90) $(OBJS) -o pure $(LIBS)

.f90.o: 
	echo "Building $@ from $<"
	$(F90) -c $(INCLUDE) $< -o $@

clean:
	/bin/rm -f pure.o pure *.pdb profile.* pure.inst.*
