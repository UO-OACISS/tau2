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
 
########### Automatically modified by the configure script ############
CONFIG_ARCH=sgi8k
CONFIG_CC=cc
CONFIG_CXX=CC
PCXX_OPT=-g
USER_OPT=-64
#######################################################################
 
############# Standard Defines ##############
CC = $(CONFIG_CC)
CXX = $(CONFIG_CXX)
DEFINES =
INSTALL = /bin/cp
SHELL = /bin/sh
LSX = .a
#############################################



############# Profiling Options #############
PROFILEOPT1           = -DPROFILING_ON    #ENDIF##PROFILE#
PROFILEOPT2         = -DSGI_TIMERS      #ENDIF##SGITIMERS#
#SGICOUNTERS#PROFILEOPT3       = -DSGI_HW_COUNTERS #ENDIF#
#PROFILECALLS#PROFILEOPT4      = -DPROFILE_CALLS   #ENDIF#
#PROFILESTATS#PROFILEOPT5      = -DPROFILE_STATS   #ENDIF#
#DEBUGPROF#PROFILEOPT6         = -DDEBUG_PROF      #ENDIF#
#KAI#PROFILEOPT7               = -DTAU_STDCXXLIB   #ENDIF#
#INTELTFLOP#PROFILEOPT8        = -DPOOMA_TFLOP     #ENDIF#
#SOL2#PROFILEOPT9              = -DNO_RTTI         #ENDIF#
#GNU#PROFILEOPT10              = -DNO_RTTI         #ENDIF#

#SGICOUNTERS#LEXTRA	       = -lperfex 	   #ENDIF#

PROFILEOPTS = $(PROFILEOPT1) $(PROFILEOPT2) $(PROFILEOPT3) $(PROFILEOPT4) \
              $(PROFILEOPT5) $(PROFILEOPT6) $(PROFILEOPT7) $(PROFILEOPT8) \
	      $(PROFILEOPT9) $(PROFILEOPT10)
##############################################

TAUROOT         = ../..

LINKER		= $(CXX)

CEXTRA 		= -w

TAUINC        	= -I$(TAUROOT)/include

CFLAGS        	= $(DEFINES) $(CEXTRA) $(USER_OPT) $(PROFILEOPTS) $(TAUINC)

LIBS 	      	= -lmpi -lm -L$(TAUROOT)/lib/$(CONFIG_ARCH) -lprof $(LEXTRA)

LDFLAGS	      	= $(USER_OPT)

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

TARGET	      	= cpi

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
