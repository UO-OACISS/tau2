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

TAUROOT=
include $(TAUROOT)/include/Makefile

CXX = $(TAU_CXX)
CC  = $(TAU_CC)

#This includes all profile options in $(PROFILEOPTS).



CEXTRA =  
############# For KAI KCC compiler, use --one_instantiation_per_object ##
#KAI#CEXTRA			= --one_per	   	   #ENDIF#
##############################################

##############################################
#PGICC#PRELINK_PHASE = $(CXX) --prelink_objects $(OBJS) #ENDIF#
#############################################

DEFINES = -DTAU_LIBRARY_SOURCE

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
		$(INCDIR)/TauFAPI.h 	 $(INCDIR)/PclLayer.h         \
		$(INCDIR)/PapiLayer.h    $(INCDIR)/JavaThreadLayer.h  \
		$(INCDIR)/OpenMPLayer.h  $(INCDIR)/TauJAPI.h	      \
		$(INCDIR)/SprocLayer.h

LDFLAGS	      = $(PCXX_OPT) 

MAKEFILE      = Makefile

OBJS	      = Profiler.o TulipTimers.o UserEvent.o FunctionInfo.o \
		RtsLayer.o RtsThread.o TauCAPI.o TauFAPI.o TauMapping.o \
		TauHooks.o $(EVENTS_O) $(THR_O) $(PCL_O) $(PAPI_O) \
		$(TAU_JAVA_O) $(OPENMP_O) $(OPARI_O) $(MULT_O) $(PLATFORM_O)

PRINT	      = pr

SRCS	      = Profiler.cpp TulipTimers.cpp UserEvent.cpp FunctionInfo.cpp \
		TauCAPI.cpp TauFAPI.cpp TauMapping.cpp TauHooks.cpp

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

RM 	      = /bin/rm -rf

TARGET	      = libtau$(TAU_OPTIONS)$(LSX)

TAU_DISABLE   = libTauDisable$(LSX)

##############################################
#MPI#MPI_WRAPPER   = libTauMpi$(TAU_OPTIONS)$(LSX)  #ENDIF#
#MPI#MPI_WRAPPER_O = TauMpi.o  #ENDIF#
#MPI#MPI_INSTALL   = $(INSTALLDEST)/libTauMpi$(TAU_OPTIONS)$(LSX)  #ENDIF#
##############################################


##############################################
# For making libTAU.so Dynamic Shared Object
##############################################
TAU_SHLIBX		= .so 		  
#HP#TAU_SHLIBX		= .sl			#ENDIF#
#GNU#AR_SHFLAGS 		= -shared  	#ENDIF#
#PGI#AR_SHFLAGS 		= -shared  	#ENDIF#
#SGICC#AR_SHFLAGS 		= -shared  	#ENDIF#
#APPLECXX#AR_SHFLAGS 		= -shared  	#ENDIF#
#SOL2#AR_SHFLAGS  		= -G 		#ENDIF#
#USE_IBMXLC#AR_SHFLAGS          = -G 		#ENDIF#
#USE_DECCXX#AR_SHFLAGS  	= -shared 		#ENDIF#
#USE_INTELCXX#AR_SHFLAGS  	= -shared 		#ENDIF#
#ACC#AR_SHFLAGS  		= -b 		#ENDIF#
TAU_ARFLAGS			= $(AR_SHFLAGS) -o 
TAU_SHLIB	= libTAUsh$(TAU_OPTIONS)$(TAU_SHLIBX)
#HITACHI# TAU_SHLIB	= #ENDIF#
#SOL2CC#EXTRA_LIBS   = -lCstd -liostream -lCrun  #ENDIF#
TAU_RANLIB = echo "Built $(TARGET)" 
#APPLECXX#TAU_RANLIB = ranlib $(TARGET) #ENDIF#

##############################################
# If JAVA or DYNINST is defined, TAU_DSO is defined
##############################################
#JAVA#TAU_DSO   		= $(TAU_SHLIB)  #ENDIF#
#DYNINST#TAU_DSO   		= $(TAU_SHLIB)  #ENDIF#
#JAVA#TAU_SHLIB_INSTALL = $(INSTALLDEST)/$(TAU_SHLIB) #ENDIF#
#JAVA#TAU_JAVA_INSTALLDIR = $(INSTALLDEST)/TAU #ENDIF#
#JAVA#TAU_JAVA_CLASS1 = $(TAU_JAVA_INSTALLDIR)/Profile.class #ENDIF#
#DYNINST# TAU_SHLIB_INSTALL = $(INSTALLDEST)/$(TAU_SHLIB) #ENDIF#
#USE_DECCXX# EXTRAOBJS=cxx_repository/*.o #ENDIF#
##############################################
TAU_DSO         = $(TAU_SHLIB)
TAU_SHLIB_INSTALL = $(INSTALLDEST)/$(TAU_SHLIB)


all:		$(TARGET) $(TAU_DISABLE) $(MPI_WRAPPER) $(TAU_DSO) $(TAUJAPI) 

$(TARGET):	$(OBJS)
		$(PRELINK_PHASE)
		$(AR) $(ARFLAGS) $(TARGET) $(OBJS) $(EXTRAOBJS)
		$(TAU_RANLIB)
		

$(TAU_SHLIB): 	$(TARGET) $(OBJS)
		$(PRELINK_PHASE)
		$(TAU_CXX) $(TAU_ARFLAGS)  $(TAU_SHLIB) $(OBJS) $(EXTRAOBJS) $(EXTRA_LIBS) $(LEXTRA) $(LEXTRA1) $(TAU_PAPI_EXTRA_FLAGS)

$(TAU_DISABLE): TauDisable.o
		$(AR) $(ARFLAGS) $(TAU_DISABLE) TauDisable.o

libTauMpi$(TAU_OPTIONS)$(LSX): 
		$(MAKE) -f Makefile.wrapper 

$(INSTALLDEST)/$(TARGET): $(TARGET)
		@echo Installing $? in $(INSTALLDEST)
		@if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)
		$(TAU_INSTALL) $(TAUROOT)/include/Makefile $(INSTALLDEST)/Makefile.tau$(TAU_OPTIONS)

$(INSTALLDEST)/$(TAU_DISABLE): $(TAU_DISABLE)
		@echo Installing $? in $(INSTALLDEST)
		@if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/$(TAU_SHLIB): $(TAU_SHLIB)
		@echo Installing $? in $(INSTALLDEST)
		@if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)
		$(RM) $(INSTALLDEST)/libTAU.so
		ln -s $@ $(INSTALLDEST)/libTAU.so

$(INSTALLDEST)/libTauMpi$(TAU_OPTIONS)$(LSX): libTauMpi$(TAU_OPTIONS)$(LSX)
		$(MAKE) -f Makefile.wrapper install

$(TAU_JAVA_INSTALLDIR)/Profile.class: Profile.class
		@echo Installing $? in $(TAU_JAVA_INSTALLDIR)
		@if [ -d $(TAU_JAVA_INSTALLDIR) ] ; then true; \
                   else mkdir $(TAU_JAVA_INSTALLDIR) ;fi
		$(TAU_INSTALL) $? $(TAU_JAVA_INSTALLDIR)
		
install:	 $(INSTALLDEST)/$(TARGET) $(INSTALLDEST)/$(TAU_DISABLE) $(MPI_INSTALL) $(TAU_SHLIB_INSTALL) $(TAU_JAVA_CLASS1)

# Rules
Profile.class	: Profile.java
	$(JDKDIR)/bin/javac Profile.java 

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

SprocLayer.o   : SprocLayer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

JavaThreadLayer.o   : JavaThreadLayer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TulipThreadLayer.o   : TulipThreadLayer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauCAPI.o   : TauCAPI.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauFAPI.o   : TauFAPI.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauJAPI.o   : TauJAPI.cpp $(HDRS) 
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauMapping.o   : TauMapping.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauHooks.o   : TauHooks.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauJava.o :	TauJava.cpp
	$(CXX) $(CXXFLAGS) -c TauJava.cpp

PclLayer.o 	: PclLayer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

PapiLayer.o 	: PapiLayer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

MultipleCounters.o : MultipleCounters.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauDisable.o 	: TauDisable.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

OpenMPLayer.o 	: OpenMPLayer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauOpari.o 	: TauOpari.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauLinuxTimers.o : TauLinuxTimers.c $(HDRS)
	gcc -c $(@:.o=.c)

$(TAUROOT)/src/Profile/TauPGIHelper.o 	: TauPGIHelper.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp) -o $@


clean : 
		$(RM) $(OBJS) $(TARGET) $(TAU_DISABLE) $(MPI_WRAPPER) $(MPI_WRAPPER_O) $(TAU_JAVA_O) $(OPENMP_O) $(TAU_DSO) $(EXTRAOBJS) $(TAUJAPI) $(FWRAPPER) *.ii *.ti ti_files cxx_repository ii_files rii_files *.a *.so

