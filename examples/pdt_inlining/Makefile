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
 
include ../../include/Makefile 
include ../Makefile
include $(TAU_MAKEFILE_MPI)

CXX		= tau_cxx.sh
CC		= tau_cc.sh
# Instrument everything!
#CFLAGS          = -optTauNoCompInst -optTau="-inline" -optKeepFiles
# Don't instrument inline functions
#CFLAGS          = -optTauNoCompInst -optTau="-noinline" -optKeepFiles
# Don't instrument inline functions or functions shorter than 5 lines
CFLAGS          = -optTauNoCompInst -optTau="-noinline -minsize 5" -optKeepFiles
LIBS            = 
LDFLAGS	      	= 
MAKEFILE      	= Makefile
PRINT	      	= pr
RM 	      	= /bin/rm -f
TARGET	      	= example
EXTRAOBJS     	= 

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(TARGET).o
	$(CXX) $(LDFLAGS) $(TARGET).o -o $@ $(LIBS)

$(TARGET).o : $(TARGET).cpp Makefile
	$(CXX) $(CFLAGS) -c $(TARGET).cpp

clean: 	
	$(RM) $(TARGET).o $(TARGET) $(TARGET).pdb $(TARGET).inst.cpp
##############################################
