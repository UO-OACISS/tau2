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
# TAU_INCLUDE  	-  Include path for tau headers
# TAU_DEFS      -  Defines that are needed for tracing and profiling only.
# And for linking add to LIBS 
# TAU_LIBS	-  TAU Tracing and Profiling library libprof.a 
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
TAUROOT=/home/users/sameer/tau2
TULIPDIR=/users/sameer/s
TAU_PREFIX_INSTALL_DIR=/home/users/sameer/tau2
TAU_OPTIONS=
#######################################################################
 
#ENABLE64BIT#ABI 	     = -64		  #ENDIF#
#ENABLEN32BIT#ABI 	     = -n32		  #ENDIF#
#ENABLE32BIT#ABI 	     = -32		  #ENDIF#

############# Standard Defines ##############
TAU_CC = $(CONFIG_CC) $(ABI) $(ISA)
TAU_CXX = $(CONFIG_CXX) $(ABI) $(ISA)
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
#STDCXXLIB#PROFILEOPT7         = -DTAU_STDCXXLIB       #ENDIF#
#INTELTFLOP#PROFILEOPT8        = -DPOOMA_TFLOP         #ENDIF#
#NORTTI#PROFILEOPT9            = -DNO_RTTI             #ENDIF#
#RTTI#PROFILEOPT9              = -DRTTI             #ENDIF#
#GNU#PROFILEOPT9           = -DNO_RTTI             #ENDIF#
#GNU#PROFILEOPT10              =             #ENDIF#
#CALLSTACK#PROFILEOPT11        = -DPROFILE_CALLSTACK   #ENDIF#
#PGI#PROFILEOPT12	       = 		       #ENDIF#
#CRAYCC#PROFILEOPT13	       = -h instantiate=used   #ENDIF#
#SGICC##PROFILEOPT14	       = -ptused 	       #ENDIF#
#PTHREAD_AVAILABLE#PROFILEOPT15 = -DPTHREADS           #ENDIF#
#TULIPTHREADS#PROFILEOPT16 = -DTULIPTHREADS            #ENDIF#
#TRACE#TRACEOPT		       = -DTRACING_ON	       #ENDIF#
#TRACE#EVENTS_O                = Tracer.o              #ENDIF#
#PTHREAD_AVAILABLE#THR_O       = PthreadLayer.o        #ENDIF#
#TULIPTHREADS#THR_O       = TulipThreadLayer.o         #ENDIF#
#TULIPTHREADS#PROFILEOPT17  = -I$(TULIPDIR)/include -I$(TULIPDIR)/Tuliplib   #ENDIF#
#SMARTS#PROFILEOPT17  = -I$(TULIPDIR)/include -I$(TULIPDIR)/lib -I$(TULIPDIR)/machine-specific/$(HOSTTYPE)  #ENDIF#
#SMARTS#PROFILEOPT18  = -DSMARTS   #ENDIF#
#KAI#PROFILEOPT19  	       = -DKAI   #ENDIF#
#PGI#PROFILEOPT19  	       = -DPGI -DTAU_DOT_H_LESS_HEADERS   #ENDIF#
#KAINOEX#PROFILEOPT20  	       = --no_exceptions   #ENDIF#
#SGICCNOEX#PROFILEOPT20  	       = -LANG:exceptions=off  #ENDIF#
#SP1#PROFILEOPT21	       = -D_POSIX_SOURCE -D_XOPEN_SOURCE -DTAU_AIX #ENDIF#

#SGICOUNTERS#LEXTRA	       = -lperfex 	       #ENDIF#
#PTHREAD_AVAILABLE#LEXTRA1     = -lpthread	       #ENDIF#
#TULIPTHREADS#LEXTRA1     = -L$(TULIPDIR)/Tuliplib  -ltulip	       #ENDIF#
#SMARTS##include $(TAUROOT)/include/makefiles/GNUmakefile-$(HOSTTYPE)  #ENDIF#
#SMARTS#LEXTRA1	    = $(LSMARTS)	#ENDIF#
#SGIMP#TAU_FORTRANLIBS	       = -lfortran -lftn       #ENDIF#
#CRAYKAI#TAU_FORTRANLIBS	       =        #ENDIF#
#PGI#TAU_FORTRANLIBS	       = $(PGI)/linux86/lib/f90main.o -lpgf90 -lpgf90rtl -lpgf90_rpm1 -lpgf902        #ENDIF#
#PGI1.7#LEXTRA = -lstd -lstrm#ENDIF#
#PGI1.7#TAUHELPER = $(TAUROOT)/src/Profile/TauPGIHelper.cpp #ENDIF#

PROFILEOPTS = $(PROFILEOPT1)  $(PROFILEOPT2)  $(PROFILEOPT3)  $(PROFILEOPT4)  \
              $(PROFILEOPT5)  $(PROFILEOPT6)  $(PROFILEOPT7)  $(PROFILEOPT8)  \
	      $(PROFILEOPT9)  $(PROFILEOPT10) $(PROFILEOPT11) $(PROFILEOPT12) \
	      $(PROFILEOPT13) $(PROFILEOPT14) $(PROFILEOPT15) $(PROFILEOPT16) \
	      $(PROFILEOPT17) $(PROFILEOPT18) $(PROFILEOPT19) $(PROFILEOPT20) \
	      $(PROFILEOPT21) $(TRACEOPT)
##############################################

############# TAU Options ####################
TAUDEFS 	= $(PROFILEOPTS) $(USER_OPT)

TAUINC        	= -I$(TAUROOT)/include

TAULIBS		= $(TAUHELPER) -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -ltau$(TAU_OPTIONS) $(LEXTRA) 

### ACL S/W requirement
TAU_DEFS	= $(TAUDEFS)

TAU_INCLUDE 	= -I$(TAUROOT)/include

TAU_LIBS	= $(TAUHELPER) -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -ltau$(TAU_OPTIONS) $(LEXTRA)

TAU_DISABLE	= $(TAUHELPER) -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -lTauDisable
## Don't include -lpthread or -lsmarts. Let app. do that. 
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
