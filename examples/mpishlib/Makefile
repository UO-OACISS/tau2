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
 
TAUROOTDIR	= ../..

include $(TAUROOTDIR)/include/Makefile

CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE)  -fPIC -g 

LIBS            = ./libring.so $(TAU_MPI_LIBS) $(TAU_SHLIBS) -lm

LDFLAGS	      	= $(USER_OPT) $(TAU_LDFLAGS)

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

TARGET	      	= cpi

EXTRAOBJS     	= 

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(TARGET).o libring.so
	$(CXX) $(LDFLAGS) $(TARGET).o -o $@ $(LIBS)

$(TARGET).o : $(TARGET).cpp
	$(CXX) $(CFLAGS) -c $(TARGET).cpp

libring.so: ring.o  
	$(CXX) $(TAU_SHFLAGS) libring.so ring.o $(TAU_MPI_LIBS) $(TAU_SHLIBS)

ring.o: ring.cpp
	$(CXX) $(CFLAGS) -c ring.cpp 

clean: 	
	$(RM) $(TARGET).o $(TARGET) ring.o libring.so 
##############################################
