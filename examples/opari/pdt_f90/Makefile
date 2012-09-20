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
#CXX		= g++

CC		= tau_cc.sh
#CC		= gcc

F90		= tau_f90.sh
#F90		= gfortran

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

$(TARGET):	$(TARGET).o mytimer.o ppm.o
	$(F90) $(LDFLAGS) $(TARGET).o mytimer.o ppm.o -o $@ $(LIBS)

$(TARGET).o : $(TARGET).f90 ppm.mod
	$(F90)  -c $(TARGET).f90

mytimer.o : mytimer.c
	$(CC) -c $<

ppm.o : ppm.f90
	$(F90)  -c $<

ppm.mod: ppm.f90
	$(F90) -c $<

clean: 	
	$(RM) *.o *.mod $(TARGET) 
##############################################
