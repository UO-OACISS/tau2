#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.acl.lanl.gov/tau		           **
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

FFLAGS        	= $(TAU_INCLUDE) $(TAU_F90_FIXED)

LIBS            = $(TAU_LIBS) $(LEXTRA1) $(TAU_FORTRANLIBS)

LDFLAGS	      	= $(USER_OPT) 

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

F90		= $(TAU_F90)  

TARGET	      	= cubes

EXTRAOBJS     	= 

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(TARGET).o  
	$(TAU_LINKER) $(LDFLAGS) $(TARGET).o  -o $@ $(LIBS)

$(TARGET).o : $(TARGET).f
	$(F90) $(FFLAGS) -c $(TARGET).f

main.o : main.C
	$(CXX) $(CFLAGS) -c main.C

clean: 	
	$(RM) $(TARGET).o $(TARGET)
##############################################
