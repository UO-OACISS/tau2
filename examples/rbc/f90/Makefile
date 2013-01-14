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

EXE = rbc

SRC = $(wildcard *.f90)
OBJ = $(SRC:.f90=.o)

F90 = tau_f90.sh
LD = tau_f90.sh

FFLAGS  = -O2 -g -optMemDbg -optKeepFiles
LDFLAGS =

RM = /bin/rm -f


##############################################

all: $(EXE)

run: $(EXE)
	tau_exec -memory_debug -T serial $(EXE)

$(EXE): $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $(EXE)

%.o: %.f90
	$(F90) $(FFLAGS) -c $< -o $@

clean: 	
	$(RM) $(OBJ) $(EXE)
	$(RM) *.inst.* *.pdb *.mod profile.*
