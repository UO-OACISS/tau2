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

MPICHDIR 	= /usr/local/packages/mpich

MPICHARCH	= LINUX/ch_p4

CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

#CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS) -I/usr/local/packages/mpich/include

CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS)  -I$(MPICHDIR)/include

#LIBS            = $(TAU_LIBS) -L/usr/local/packages/mpich/lib/LINUX/ch_p4 -lmpich $(LEXTRA1)

LIBS            = $(TAU_LIBS) -L$(MPICHDIR)/lib/$(MPICHARCH) -lmpich $(LEXTRA1)

#LIBS            = $(TAU_LIBS) -lmpi

LDFLAGS	      	= $(USER_OPT)

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

TARGET	      	= matrix

EXTRAOBJS     	= 

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(TARGET).o
	$(CXX) $(LDFLAGS) $(TARGET).o -o $@ $(LIBS)

$(TARGET).o : $(TARGET).cpp
	$(CXX) $(CFLAGS) -c $(TARGET).cpp

clean: 	
	$(RM) $(TARGET).o $(TARGET)
##############################################
