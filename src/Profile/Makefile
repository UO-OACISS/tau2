#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.acl.lanl.gov/tau		           **
#****************************************************************************
#*    Copyright 1997-99  						   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#*    Research Center Juelich, ZAM Germany				   **
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

TAUROOT=/home/users/sameer/tau2
include $(TAUROOT)/include/Makefile

CXX = $(TAU_CXX)
CC  = $(TAU_CC)

#This includes all profile options in $(PROFILEOPTS).


CEXTRA = 

##############################################
#PGICC#PRELINK_PHASE = $(CXX) --prelink_objects $(OBJS) #ENDIF#
#############################################



LINKER	= $(TAU_CC)

INSTALLDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

TAUINC	      = -I$(TAUROOT)/include 

CXXFLAGS        = $(DEFINES) $(CEXTRA) $(USER_OPT) $(PROFILEOPTS) $(TAUINC)

INCDIR	      = $(TAUROOT)/include/Profile

HDRS	      = $(INCDIR)/Profiler.h 	 $(INCDIR)/ProfileGroups.h    \
		$(INCDIR)/TauAPI.h 	 $(INCDIR)/ProfileHeaders.h   \
		$(INCDIR)/PthreadLayer.h $(INCDIR)/RtsLayer.h         \
		$(INCDIR)/FunctionInfo.h $(INCDIR)/UserEvent.h        \
		$(INCDIR)/TauCAPI.h 	 $(INCDIR)/TulipTimers.h      \
	        $(INCDIR)/bstring.h	 $(INCDIR)/TulipThreadLayer.h \
		$(INCDIR)/TauFAPI.h 	 $(INCDIR)/PclLayer.h

LDFLAGS	      = $(PCXX_OPT)

MAKEFILE      = Makefile

OBJS	      = Profiler.o TulipTimers.o UserEvent.o FunctionInfo.o \
		RtsLayer.o RtsThread.o TauCAPI.o TauFAPI.o TauMapping.o \
		$(EVENTS_O) $(THR_O) $(PCL_O)

PRINT	      = pr

SRCS	      = Profiler.cpp TulipTimers.cpp UserEvent.cpp FunctionInfo.cpp \
		TauCAPI.cpp TauFAPI.cpp TauMapping.cpp 

AR	      = ar

ARFLAGS	      = rcv 
############# For KAI KCC compiler, replace AR accordingly #######
#KAI#AR			= $(CXX) 		   	   #ENDIF#
#KAI#ARFLAGS			= --one_per -o		 	   #ENDIF#
##################################################################

############# For SGI CC compiler, replace AR accordingly #######
#SGICC#AR			= $(CXX) -ar 		   #ENDIF#
#SGICC#ARFLAGS			= -o			   #ENDIF#
##################################################################

############# For FUJITSU FCC compiler, replace AR accordingly #######
#FUJITSU#AR			= $(CXX) --ar 		   #ENDIF#
#FUJITSU#ARFLAGS	        = -o			   #ENDIF#
##################################################################

############# For SUN CC compiler, replace AR accordingly #######
#SOL2CC#AR			= $(CXX) -xar 		   #ENDIF#
#SOL2CC#ARFLAGS			= -o			   #ENDIF#
##################################################################

RM 	      = /bin/rm -f

TARGET	      = libtau$(TAU_OPTIONS)$(LSX)

TAU_DISABLE   = libTauDisable$(LSX)

##############################################
#MPI#MPI_WRAPPER   = libTauMpi$(LSX)  #ENDIF#
#MPI#MPI_WRAPPER_O = TauMpi.o  #ENDIF#
#MPI#MPI_INSTALL   = $(INSTALLDEST)/libTauMpi$(LSX)  #ENDIF#
##############################################

##############################################
TAU_SHLIBX		= .so 		  
#JAVA#TAU_JAVALIB   		= libTAU$(TAU_SHLIBX)  #ENDIF#
#JAVA#TAU_JAVA_O 		= TauJava.o  	   #ENDIF#
#JAVA#TAU_JAVA_INSTALL  		= $(INSTALLDEST)/$(TAU_JAVALIB)  #ENDIF#
#GNU#AR_SHFLAGS 		= -shared  	#ENDIF#
#PGI#AR_SHFLAGS 		= -shared  	#ENDIF#
#SOL2#AR_SHFLAGS  		= -G	        #ENDIF#
TAU_ARFLAGS			= $(AR_SHFLAGS) -o 
##############################################

all:		$(TARGET) $(TAU_DISABLE) $(MPI_WRAPPER) $(TAU_JAVALIB)

$(TARGET):	$(OBJS)
		$(PRELINK_PHASE)
		$(AR) $(ARFLAGS) $(TARGET) $(OBJS) $(EXTRAOBJS)

$(TAU_DISABLE): TauDisable.o
		$(AR) $(ARFLAGS) $(TAU_DISABLE) TauDisable.o

libTauMpi$(LSX): 
		$(MAKE) -f Makefile.wrapper 

libTAU$(TAU_SHLIBX): $(TARGET) 
		$(MAKE) -f Makefile.java ARFLAGS='$(TAU_ARFLAGS)' PRELINK_PHASE='$(PRELINK_PHASE)'
 

$(INSTALLDEST)/$(TARGET): $(TARGET)
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(INSTALL) $? $(INSTALLDEST)
		$(INSTALL) $(TAUROOT)/include/Makefile $(INSTALLDEST)/Makefile.tau$(TAU_OPTIONS)

$(INSTALLDEST)/$(TAU_DISABLE): $(TAU_DISABLE)
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(INSTALL) $? $(INSTALLDEST)

		
$(INSTALLDEST)/libTauMpi$(LSX): libTauMpi$(LSX)
		$(MAKE) -f Makefile.wrapper install

$(INSTALLDEST)/libTAU$(TAU_SHLIBX): libTAU$(TAU_SHLIBX)
		$(MAKE) -f Makefile.java ARFLAGS='$(TAU_ARFLAGS)' PRELINK_PHASE='$(PRELINK_PHASE)' install

install:	 $(INSTALLDEST)/$(TARGET) $(INSTALLDEST)/$(TAU_DISABLE) $(MPI_INSTALL) $(TAU_JAVA_INSTALL)

# Rules
Profiler.o 	: Profiler.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TulipTimers.o 	: TulipTimers.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

UserEvent.o 	: UserEvent.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

FunctionInfo.o 	: FunctionInfo.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

RtsLayer.o 	: RtsLayer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

RtsThread.o 	: RtsThread.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

pcxx_events.o   : pcxx_events.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

Tracer.o   	: Tracer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

PthreadLayer.o   : PthreadLayer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TulipThreadLayer.o   : TulipThreadLayer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauCAPI.o   : TauCAPI.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauFAPI.o   : TauFAPI.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauMapping.o   : TauMapping.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

PclLayer.o 	: PclLayer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauDisable.o 	: TauDisable.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

$(TAUROOT)/src/Profile/TauPGIHelper.o 	: TauPGIHelper.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp) -o $@


clean : 
		$(RM) $(OBJS) $(TARGET) $(TAU_DISABLE) $(MPI_WRAPPER) $(MPI_WRAPPER_O) $(TAU_JAVA_O) $(TAU_JAVALIB) $(EXTRAOBJS) *.ii *.ti ti_files/*

