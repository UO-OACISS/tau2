#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/tau	           **
#****************************************************************************
#*    Copyright 1997  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
#######################################################################
##                  pC++/Sage++  Copyright (C) 1993,1995             ##
##  Indiana University  University of Oregon  University of Rennes   ##
#######################################################################


TAU_MAKEFILE ?=../../../include/Makefile

include $(TAU_MAKEFILE)

CXX		= TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_cxx.sh -optVerbose -optCompInst -optShared

PRINT	      	= pr

RM 	      	= /bin/rm -rf

TARGET	      	= program

OBJS     	= main.o
CFLAGS=-g -O0 $(TAU_OPENMP_OPTION) -I.
LDFLAGS=-g $(TAU_OPENMP_OPTION)

##############################################

all:		$(TARGET)	$(TARGET)_mpi

install: 	$(TARGET)	$(TARGET)_mpi

$(TARGET):	$(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

$(TARGET)_mpi:	main_mpi.o
	$(CXX) $(LDFLAGS) main_mpi.o -o $@ $(LIBS)

main.o: main.cxx headers1/A.h headers2/B.h
	export TAU_COMPILER_VERBOSE=1
	$(CXX) $(CFLAGS) -c $< -o $@

main_mpi.o: main.cxx headers1/A.h headers2/B.h
	export TAU_COMPILER_VERBOSE=1
	$(CXX) $(CFLAGS) -c $< -o $@ -DUSE_MPI

clean:
	$(RM) $(OBJS) $(TARGET) profile.* *.trc *.edf *.z MULT* *.inst.* *.pdb Comp_gnu.o *.pomp.c *.opari.inc pompregions.* *.output *.error *.cobaltlog
##############################################
