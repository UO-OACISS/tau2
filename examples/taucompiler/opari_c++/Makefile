
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

CFLAGS 		= 

FFLAGS        	= 

LIBS            =  

LDFLAGS	      	= $(USER_OPT) 

MAKEFILE      	= Makefile

RM 	      	= /bin/rm -f

TARGET	      	= omp_hello

##############################################

all:		$(TARGET) 

install: 	$(TARGET)

$(TARGET):	$(TARGET).o
	$(CXX) -o omp_hello omp_hello.o

$(TARGET).o : $(TARGET).cc 
	$(CXX) -c $< 

clean: 	
	@echo "*********************************************************"
	@echo "Cleaning... :"
	$(RM) -r $(TARGET).o $(TARGET) 
##############################################

