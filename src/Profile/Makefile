#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.acl.lanl.gov/tau		           **
#****************************************************************************
#*    Copyright 1997  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
#****************************************************************************
#**      File            : Makefile                                        **
#**      Description     : TAU Profiling Package                           **
#**      Author          : Sameer Shende                                   **
#**      Contact         : sameer@cs.uoregon.edu        		   **
#**      Flags           : Compile with                                    **
#**                        -DPROFILING_ON to enable profiling (ESSENTIAL)  **
#**                        -DPROFILE_STATS for Std. Deviation of Excl Time **
#**                        -DSGI_HW_COUNTERS for using SGI counters        **
#**                        -DPROFILE_CALLS  for trace of each invocation   **
#**                        -DSGI_TIMERS  for SGI fast nanosecs timer       **
#**                        -DPOOMA_TFLOP for Intel Teraflop at SNL/NM      **
#**                        -DUSE_STDCXXLIB for KCC compiler                **
#**                        -DDEBUG_PROF  for internal debugging messages   **
#**			   These flags are set by configure script	   **
#**      Documentation   : See http://www.acl.lanl.gov/tau	           **
#****************************************************************************

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
#DEBUGPROF#PROFILEOPT6	       = -DDEBUG_PROF	   #ENDIF#
#KAI#PROFILEOPT7	       = -DTAU_STDCXXLIB   #ENDIF#
#INTELTFLOP#PROFILEOPT8        = -DPOOMA_TFLOP	   #ENDIF#
#SOL2CC#PROFILEOPT9	       = -DNO_RTTI	   #ENDIF#
#GNU#PROFILEOPT10	       = -DNO_RTTI	   #ENDIF#

PROFILEOPTS = $(PROFILEOPT1) $(PROFILEOPT2) $(PROFILEOPT3) $(PROFILEOPT4) \
	      $(PROFILEOPT5) $(PROFILEOPT6) $(PROFILEOPT7) $(PROFILEOPT8) \
	      $(PROFILEOPT9) $(PROFILEOPT10)
##############################################
#KAI#AR			= $(CONFIG_CXX)     #ENDIF#
#KAI#ARFLAGS			= -o		    #ENDIF#
#SOL2CC#EXTRAOBJS		= Templates.DB/*.o  #ENDIF#
#############################################


LINKER	= $(CC)

TAUROOT		= ../..

INSTALLDEST	= $(TAUROOT)/lib/$(CONFIG_ARCH)

CEXTRA = 

TAUINC	      = -I$(TAUROOT)/include 

CFLAGS        = $(DEFINES) $(CEXTRA) $(USER_OPT) $(PROFILEOPTS) $(TAUINC)

HDRS	      = $(TAUROOT)/include/Profile/Profiler.h \
	 	$(TAUROOT)/include/Profile/TulipTimers.h \
		$(TAUROOT)/include/Profile/bstring.h

LDFLAGS	      = $(PCXX_OPT)

MAKEFILE      = Makefile

OBJS	      = Profiler.o TulipTimers.o

PRINT	      = pr

SRCS	      = Profiler.cpp TulipTimers.cpp

AR	      = ar

ARFLAGS	      = rcv 
############# For KAI KCC compiler, replace AR accordingly #######
#KAI#AR			= $(CONFIG_CXX) 		   #ENDIF#
#KAI#ARFLAGS			= -o				   #ENDIF#
##################################################################

RM 	      = /bin/rm -f

TARGET	      = libprof$(LSX)

all:		$(TARGET)	

$(TARGET):	$(OBJS)
		$(AR) $(ARFLAGS) $(TARGET) $(OBJS) $(EXTRAOBJS)

$(INSTALLDEST)/$(TARGET): $(TARGET)
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(INSTALL) $? $(INSTALLDEST)

install:	$(INSTALLDEST)/$(TARGET) 

program:        $(PROGRAM)


Profiler.o 	: Profiler.cpp $(HDRS)
	$(CXX) $(CFLAGS) -c Profiler.cpp  

TulipTimers.o 	: TulipTimers.cpp $(HDRS)
	$(CXX) $(CFLAGS) -c TulipTimers.cpp

clean : 
		$(RM) $(OBJS) $(TARGET) $(EXTRAOBJS)

