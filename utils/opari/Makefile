#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/paracomp/tau    **
#****************************************************************************
#*    Copyright 1997  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
#######################################################################
##                  pC++/Sage++  Copyright (C) 1993,1995             ##
##  Indiana University  University of Oregon  University of Rennes   ##
#######################################################################
 
include ../../include/Makefile
########### Automatically modified by the configure script ############


COMP=$(TAU_CXX_FE) $(ABI) $(ISA)
RM=/bin/rm -f

FILES=opari.cc process_c.cc process_f.cc process_omp.cc ompragma.cc ompragma_c.cc ompragma_f.cc ompregion.cc handler.cc 

all: opari
install: opari

opari: $(FILES)
	$(COMP) $(FILES) -o opari

clean:
	$(RM) opari
