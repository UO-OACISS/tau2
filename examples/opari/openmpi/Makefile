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

CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS)  $(TAU_MPI_INCLUDE)

LIBS            = $(TAU_MPI_LIBS) $(TAU_LIBS) $(LEXTRA1) $(EXTRALIBS)  -lm

LDFLAGS	      	= $(USER_OPT)  $(TAU_LDFLAGS)

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

TARGET	      	= stommel

EXTRAOBJS     	= 

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(TARGET).mod.o opari.tab.o
	$(CXX) $(LDFLAGS) $(TARGET).mod.o opari.tab.o -o $@ $(LIBS)

$(TARGET).mod.o : $(TARGET).mod.c
	$(CC) $(CFLAGS) -c $(TARGET).mod.c

$(TARGET).mod.c : $(TARGET).c
	$(RM) opari.rc
	$(OPARIDIR)/tool/opari -table opari.tab.c $(TARGET).c

opari.tab.o: opari.tab.c $(TARGET).mod.c
	$(CC) $(CFLAGS) -c opari.tab.c

clean: 	
	$(RM) $(TARGET).mod.o $(TARGET) $(TARGET).mod.c opari.tab.c *.pv *.trc *.edf profile.* opari.rc *.inc opari.tab.o
##############################################
