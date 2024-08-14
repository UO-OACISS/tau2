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



CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

HDF5_ROOT	= /storage/users/jalcaraz/spack/opt/spack/linux-ubuntu22.04-icelake/gcc-11.4.0/hdf5-1.14.3-niejd7ngst5d42hij6b34767qh6v5qwg/

CFLAGS		= -I$(HDF5_ROOT)/include
#CFLAGS        	= $(TAU_INCLUDE) $(TAU_DEFS)
#CFLAGS        	= 

LIBS		= -L$(HDF5_ROOT)/lib -lhdf5
#LIBS 	      	= $(TAU_LIBS) 
#LIBS 	      	= $(TAUROOTDIR)/src/Profile/TauHooks.o $(TAU_LIBS) -ldl

LDFLAGS	      	= $(USER_OPT)

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

TARGET	      	= h5ex_d_hyper

EXTRAOBJS     	= 

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(TARGET).o
	$(CC) $(LDFLAGS) $(TARGET).o -o $@ $(LIBS)

$(TARGET).o : $(TARGET).c
	$(CC) $(CFLAGS) -c $(TARGET).c

clean: 	
	$(RM) $(TARGET).o $(TARGET) profile.* 
##############################################
