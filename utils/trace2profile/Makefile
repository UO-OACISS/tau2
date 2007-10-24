#****************************************************************************
#*                      TAU Portable Profiling Package                     **
#*                      http://www.cs.uoregon.edu/research/paracomp/tau    **
#****************************************************************************
#*    Copyright 1997                                                       **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
#######################################################################
##                  pC++/Sage++  Copyright (C) 1993,1995             ##
##  Indiana University  University of Oregon  University of Rennes   ##
#######################################################################

include ../../include/Makefile
########### Automatically modified by the configure script ############



#SGICC#LEXTRA = -lCio #ENDIF#




CONFIG_ARCH=default
CONFIG_CC=gcc
CONFIG_CXX=g++
CXX=$(CONFIG_CXX)  $(ABI) $(ISA)
HEADERS=
LIBS=$(TAU_TRACE_INPUT_LIB)
CFLAGS=$(TAU_INCLUDE) -I.
RM=/bin/rm -f

#The following are variables for all the files and their associated variations.
HANDLERS=handlers.o handlers.*.~*
READTRACE=readTrace.o readTrace.*.~*
TRACE2PROFILE=trace2profile.o trace2profile.*.~*
MAKEFILE=Makefile.~*

all:    trace2profile

install:       trace2profile
	
trace2profile: handlers.o readTrace.o trace2profile.o
	$(CXX) $(CFLAGS) -o $@ handlers.o readTrace.o trace2profile.o $(LIBS)
	
handlers.o: handlers.h trace2profile.h handlers.cpp
	$(CXX) $(CFLAGS) -c handlers.cpp

readTrace.o: handlers.h trace2profile.h readTrace.cpp
	$(CXX) $(CFLAGS) -c readTrace.cpp
	
trace2profile.o: trace2profile.h trace2profile.cpp
	$(CXX) $(CFLAGS) -c trace2profile.cpp

 	
clean:
	$(RM) $(TRACE2PROFILE) $(HANDLERS) $(READTRACE) $(MAKEFILE)

