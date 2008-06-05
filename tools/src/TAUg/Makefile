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
 
CONFIG_ARCH=x86_64
TAUROOT=/home/khuck/tau2

include $(TAUROOT)/include/Makefile

CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE)

LIBS            = $(TAU_MPI_LIBS) $(TAU_LIBS) -lm

LDFLAGS	      	= $(USER_OPT) $(TAU_LDFLAGS)

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

TARGET	      	= libtauglobal.a

HEADER	      	= TauGlobal.h
HEADERF	      	= TauGlobalf.h

EXTRAOBJS     	= 
ARFLAGS       = rcv

##############################################

all:		$(TARGET)

install: 	$(TARGET)
	cp $(TARGET) $(TAUROOT)/$(CONFIG_ARCH)/lib/.
	cp TauGlobal.h $(TAUROOT)/include/.
	cp TauGlobalf.h $(TAUROOT)/include/.

$(TARGET):	TauGlobal.o TauGlobalf.o
	$(TAU_AR) $(ARFLAGS) $(TARGET) TauGlobal.o TauGlobalf.o

TauGlobal.o : TauGlobal.cpp
	$(CXX) $(CFLAGS) -c TauGlobal.cpp

TauGlobalf.o : TauGlobalf.cpp
	$(CXX) $(CFLAGS) -c TauGlobalf.cpp

clean: 	
	$(RM) $(TARGET).o $(TARGET) TauGlobal.o TauGlobalf.o
##############################################
