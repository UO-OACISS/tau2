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
 
include ../../../include/Makefile 
include ../../Makefile



CXX		= $(TAU_CXX) 

CC		= $(TAU_CC) 

FFLAGS        	= 

LIBS            =  $(TAU_LIBS)  $(TAU_FORTRANLIBS)  $(LEXTRA1)
#LIBS            = $(TAU_DISABLE)  $(TAU_FORTRANLIBS)

LDFLAGS	      	= $(USER_OPT) $(TAU_OPENMP_OPTION)

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

F90		= $(TAU_F90) $(TAU_F90_SUFFIX)

TARGET	      	= mandel

EXTRAOBJS     	= 

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	ppm.o $(TARGET).o mytimer.o
	$(TAU_LINKER) $(LDFLAGS) $(TARGET).o ppm.o mytimer.o -o $@ $(LIBS)

$(TARGET).o : $(TARGET).f90
	$(F90) $(FFLAGS) -c $(TARGET).f90

ppm.o : ppm.f90
	$(F90) $(FFLAGS) -c $<

mytimer.o : mytimer.c
	$(CC) -c $<

clean: 	
	$(RM) $(TARGET).o $(TARGET) mytimer.o ppm.o *.mod
##############################################
