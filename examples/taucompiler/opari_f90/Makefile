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
 
TAUROOTDIR	= ../../..

include $(TAUROOTDIR)/include/Makefile

F90		= $(TAU_COMPILER) $(TAU_F90) 

CXX		= $(TAU_COMPILER) $(TAU_CXX) 

CC		= $(TAU_COMPILER) $(TAU_CC) 

CFLAGS 		= 

FFLAGS        	= $(TAU_F90_SUFFIX) 

LIBS            =  

LDFLAGS	      	= $(USER_OPT) 

MAKEFILE      	= Makefile

RM 	      	= /bin/rm -f

TARGET	      	= mandel

##############################################

all:		$(TARGET) 

install: 	$(TARGET)

$(TARGET):	ppm.o $(TARGET).o mytimer.o 
	@echo "*********************************************************"
	@echo "LINKING: "
	$(F90) $(LDFLAGS) $(TARGET).o ppm.o mytimer.o -o $@ $(LIBS)

$(TARGET).o : $(TARGET).f90 ppm.o 
	@echo "*********************************************************"
	@echo "Creating $(TARGET).o:"
	$(F90) $(FFLAGS) -c $< 

ppm.o : ppm.f90
	@echo "*********************************************************"
	@echo "Creating ppm.o: "
	$(F90) $(FFLAGS) -c $< -o $@

mytimer.o : mytimer.c
	@echo "*********************************************************"
	@echo "Creating mytimer.o:"
	$(CC) $(CFLAGS) -c $< -o $@

clean: 	
	@echo "*********************************************************"
	@echo "Cleaning... :"
	$(RM) -r $(TARGET).o $(TARGET) mytimer.o ppm.o *.mod* 
##############################################

