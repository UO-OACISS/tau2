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
 
#######################################################################
# This is a sample Makefile that contains the Profiling and Tracing 
# options. Makefiles of other applications and libraries (not included 
# in this distribution) should include this Makefile.
# It defines the following variables that should be added to CFLAGS
# TAUINC  	-  Include path for tau headers
# TAUDEFS	-  Defines that are needed for tracing and profiling only.
# And for linking add to LIBS 
# TAULIBS	-  TAU Tracing and Profiling library libprof.a 
# 
# When the user needs to turn off tracing and profiling and run the 
# application without any runtime overhead of instrumentation, simply
# remove TAUDEFS and TAULIBS from CFLAGS and LIBS respectively but keep
# TAUINC.
#######################################################################

########### Automatically modified by the configure script ############
CONFIG_ARCH=default
CONFIG_CC=gcc
CONFIG_CXX=g++
PCXX_OPT=-g
USER_OPT=-g
TAUROOT=/users/sameer/rs/tau2
#######################################################################
 
#MIPSR8K##ABI 	     = -64		  #ENDIF#
#MIPSR8K#ISA          = -mips4             #ENDIF#

############# Standard Defines ##############
CC = $(CONFIG_CC) $(ABI) $(ISA)
CXX = $(CONFIG_CXX) $(ABI) $(ISA)
DEFINES =
INSTALL = /bin/cp
SHELL = /bin/sh
LSX = .a
#############################################



############# Profiling Options #############
#PROFILE#PROFILEOPT1           = -DPROFILING_ON        #ENDIF#
#SGITIMERS#PROFILEOPT2         = -DSGI_TIMERS          #ENDIF#
#SGICOUNTERS#PROFILEOPT3       = -DSGI_HW_COUNTERS     #ENDIF#
#PROFILECALLS#PROFILEOPT4      = -DPROFILE_CALLS       #ENDIF#
#PROFILESTATS#PROFILEOPT5      = -DPROFILE_STATS       #ENDIF#
#DEBUGPROF#PROFILEOPT6         = -DDEBUG_PROF          #ENDIF#
#KAI#PROFILEOPT7               = -DTAU_STDCXXLIB       #ENDIF#
#INTELTFLOP#PROFILEOPT8        = -DPOOMA_TFLOP         #ENDIF#
#SOL2#PROFILEOPT9              = -DNO_RTTI             #ENDIF#
#GNU#PROFILEOPT10              = -DNO_RTTI             #ENDIF#
#CALLSTACK#PROFILEOPT11        = -DPROFILE_CALLSTACK   #ENDIF#
#PGI#PROFILEOPT12	       = 		       #ENDIF#
#CRAYCC#PROFILEOPT13	       = -h instantiation=used #ENDIF#
#SGICC##PROFILEOPT14	       = -ptused 	       #ENDIF#
#PTHREAD_AVAILABLE#PROFILEOPT15 = -DPTHREADS           #ENDIF#
#TRACE#TRACEOPT		       = -DTRACING_ON	       #ENDIF#
#TRACE#EVENTS_O                = pcxx_events.o         #ENDIF#
#PTHREAD_AVAILABLE#THR_O       = PthreadLayer.o           #ENDIF#

#SGICOUNTERS#LEXTRA	       = -lperfex 	       #ENDIF#
#PTHREAD_AVAILABLE#LEXTRA      += -lpthread	       #ENDIF#

PROFILEOPTS = $(PROFILEOPT1)  $(PROFILEOPT2)  $(PROFILEOPT3)  $(PROFILEOPT4)  \
              $(PROFILEOPT5)  $(PROFILEOPT6)  $(PROFILEOPT7)  $(PROFILEOPT8)  \
	      $(PROFILEOPT9)  $(PROFILEOPT10) $(PROFILEOPT11) $(PROFILEOPT12) \
	      $(PROFILEOPT13) $(PROFILEOPT14) $(PROFILEOPT15) $(TRACEOPT)
##############################################

############# TAU Options ####################
TAUDEFS 	= $(PROFILEOPTS) $(USER_OPT)

TAUINC        	= -I$(TAUROOT)/include

TAULIBS		= -L$(TAUROOT)/lib/$(CONFIG_ARCH) -lprof $(LEXTRA)

#############################################
# These options could be included in the application Makefile as 
#CFLAGS        	= $(USER_OPT) $(TAUDEFS) $(TAUINC)
#
#LIBS 	      	= $(TAULIBS)
#
# To run the application without Profiling/Tracing use
#CFLAGS		= $(TAUINC)   
# Don't use TAUDEFS but do include TAUINC
# Also ignore TAULIBS when Profiling/Tracing is not used.
#############################################
