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

CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS) -DSTD_COMPLEX

LIBS            = $(TAU_LIBS) 

LDFLAGS	      	= $(CFLAGS)

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

TARGET	      	= mandel

EXTRAOBJS     	= 

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(TARGET).mod.o mytimer.o ppmwrite.o opari.tab.o
	$(CXX) $(LDFLAGS) $(TARGET).mod.o opari.tab.o mytimer.o ppmwrite.o -o $@ $(LIBS)

$(TARGET).mod.o : $(TARGET).mod.cpp
	$(CXX) $(CFLAGS) -c $(TARGET).mod.cpp

$(TARGET).mod.cpp : $(TARGET).cpp
	$(RM) opari.rc
	$(TAU_OPARI_TOOL) -table opari.tab.c $(TARGET).cpp

opari.tab.o: opari.tab.c $(TARGET).mod.cpp 
	$(CC) $(CFLAGS) -c opari.tab.c

mytimer.o : mytimer.c
	$(CC) -c $<

ppmwrite.o : ppmwrite.cpp
	$(CXX) $(CFLAGS) -c $<

clean: 	
	$(RM) $(TARGET).mod.o $(TARGET).cpp.opari.inc $(TARGET).mod.cpp \
	$(TARGET) ppmwrite.o mytimer.o opari.rc opari.tab.c opari.tab.o 
##############################################
