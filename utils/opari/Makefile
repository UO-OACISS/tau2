#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/paracomp/tau    **
#****************************************************************************
#*    Copyright 1997-2009					   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************

include ../../include/Makefile

COMP=$(TAU_CXX_FE) $(ABI) $(ISA)
RM=/bin/rm -f

SOURCES=opari.cc process_c.cc process_f.cc process_omp.cc ompragma.cc ompragma_c.cc ompragma_f.cc ompregion.cc handler.cc 

OBJECTS=opari.o process_c.o process_f.o process_omp.o ompragma.o ompragma_c.o ompragma_f.o ompregion.o handler.o 


all: opari
install: opari

opari: $(OBJECTS)
	$(COMP) $(OBJECTS) -o opari

.cc.o:
	$(COMP) -c -o $@ $<

clean:
	$(RM) opari $(OBJECTS)
