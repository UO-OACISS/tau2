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
 
CXX		= tau_cxx.sh

CC		= tau_cc.sh

CFLAGS          = 

LIBS            = 

LDFLAGS	      	= $(CFLAGS)

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

TARGET	      	= mandel

EXTRAOBJS     	= 

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(TARGET).o mytimer.o ppmwrite.o 
	$(CXX) $(LDFLAGS) $(TARGET).o mytimer.o ppmwrite.o -o $@ $(LIBS)

$(TARGET).o : $(TARGET).cpp
	$(CXX) $(CFLAGS) -c $(TARGET).cpp

mytimer.o : mytimer.c
	$(CC) -c $<

ppmwrite.o : ppmwrite.cpp
	$(CXX) $(CFLAGS) -c $<

clean: 	
	$(RM) *.o $(TARGET) 
##############################################
