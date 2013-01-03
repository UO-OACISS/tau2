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

SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)

CXX = tau_cxx.sh
LD = $(CXX)

CXXFLAGS = -optMemDbg -O2 -g -optKeepFiles
LDFLAGS =

MAKEFILE      	= Makefile

RM 	      	= /bin/rm -f


##############################################

all: $(EXE)

$(EXE): $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $(EXE)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean: 	
	$(RM) $(OBJ) $(EXE)
	$(RM) *.inst.* *.pdb profile.*
