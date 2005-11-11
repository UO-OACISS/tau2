#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/tau	           **
#****************************************************************************
#*    Copyright 2004  						   	   **
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

CFLAGS        	= $(TAU_INCLUDE) $(TAU_INCLUDE_MEMORY) $(TAU_DEFS) -UTAU_MPI 

LIBS 	      	= $(TAU_LIBS) 

LDFLAGS	      	= $(USER_OPT)

RM 	      	= /bin/rm -f

TARGET	      	= klargest

OBJS		= main.o select.o

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

%.o:%.cpp
	$(CXX) $(CFLAGS) -c $<

clean: 	
	$(RM) $(OBJS) $(TARGET)
##############################################
