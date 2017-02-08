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
 
CC      =gcc
CXX      =g++
PRINT	      	=pr
RM 	      	=/bin/rm -rf
TARGET	      	=openmp_test
OBJS     	=$(TARGET).o
CXXFLAGS=-g -O3 -fopenmp $(shell PKG_CONFIG_PATH=../../../x86_64/lib/pkgconfig pkg-config --cflags apex)
LDFLAGS=-g -O3
LIBS=$(shell PKG_CONFIG_PATH=../../../x86_64/lib/pkgconfig pkg-config --libs apex)

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(OBJS) 
	$(CXX) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

$(TARGET).o: $(TARGET).c
	$(CC) $(CXXFLAGS) -c $< -o $@

clean: 	
	$(RM) $(OBJS) $(TARGET) profile.* *.trc *.edf *.z MULT* *.inst.* *.pdb Comp_gnu.o *.pomp.c *.opari.inc pompregions.* *.output *.error *.cobaltlog
##############################################
