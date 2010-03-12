#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/tau	           **
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
#**	 Contact	 : tau-team@cs.uoregon.edu  		 	   **
#**      Documentation   : See http://www.cs.uoregon.edu/research/tau       **
#****************************************************************************

TAUROOT=
include $(TAUROOT)/include/Makefile

CXX = $(TAU_CXX)
CC  = $(TAU_CC)

#This includes all profile options in $(PROFILEOPTS).


BFDINCLUDE=
BFDLINK=

#TAU_PERFSUITE#PERFSUITE_OPT=-DTAU_PERFSUITE #ENDIF#
PERFSUITEINCLUDE=
PERFSUITELINK=


#HPCTOOLKIT#HPCTOOLKIT_OPT=-DTAU_USE_HPCTOOLKIT #ENDIF#
HPCTOOLKIT_INCLUDE=
HPCTOOLKIT_LINK=



CEXTRA =  
############# For KAI KCC compiler, use --one_instantiation_per_object ##
#KAI#CEXTRA			= --one_per	   	   #ENDIF#
##############################################

##############################################
#PGICC#PRELINK_PHASE = $(CXX) --prelink_objects $(OBJS) #ENDIF#
#PGINOPRELINK#PRELINK_PHASE = #ENDIF#
#############################################

DEFINES = -DTAU_LIBRARY_SOURCE


LINKER	= $(TAU_CC)

INSTALLDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

TAUINC	      = -I$(TAUROOT)/include 

CXXFLAGS      = $(DEFINES) $(CEXTRA) $(USER_OPT) $(TAUINC) $(PROFILEOPTS) $(TAU_INTERNAL_FLAGS) $(PERFSUITE_OPT) $(PERFSUITEINCLUDE)  $(HPCTOOLKIT_OPT) $(HPCTOOLKIT_INCLUDE)

UPCFLAGS        = $(TAUINC) $(PROFILEOPTS) 

CFLAGS        = $(TAU_MPI_INCLUDE) $(CEXTRA) $(USER_OPT) $(TAUINC) $(PROFILEOPTS) $(TAU_INTERNAL_FLAGS)   

INCDIR	      = $(TAUROOT)/include/Profile

HDRS	      = $(INCDIR)/Profiler.h 	 $(INCDIR)/ProfileGroups.h    \
		$(INCDIR)/TauAPI.h 	 $(INCDIR)/ProfileHeaders.h   \
		$(INCDIR)/PthreadLayer.h $(INCDIR)/RtsLayer.h         \
		$(INCDIR)/FunctionInfo.h $(INCDIR)/UserEvent.h        \
		$(INCDIR)/TauCppAPI.h 	 $(INCDIR)/bstring.h	      \
		$(INCDIR)/TauFAPI.h 	 $(INCDIR)/TauMetrics.h       \
		$(INCDIR)/PapiLayer.h    $(INCDIR)/JavaThreadLayer.h  \
		$(INCDIR)/OpenMPLayer.h  $(INCDIR)/TauJAPI.h	      \
		$(INCDIR)/SprocLayer.h   $(INCDIR)/PyGroups.h         \
		$(INCDIR)/PyExceptions.h $(INCDIR)/PyDatabase.h       \
		$(INCDIR)/PyBindings.h   $(INCDIR)/PyTimer.h          \
		$(INCDIR)/RenciSTFF.h    $(INCDIR)/PapiThreadLayer.h  \
		$(INCDIR)/TauEnv.h	 $(INCDIR)/TauInit.h          \
		$(INCDIR)/TauTrace.h     $(INCDIR)/TauSilc.h          \
		$(TAUROOT)/include/tau_internal.h $(INCDIR)/TauSampling.h \
		$(INCDIR)/TauGASP.h

LDFLAGS	      =

MAKEFILE      = Makefile

PGIACC_O=
#PGIACC#PGIACC_O = PgiAcc.o #ENDIF#
#PGI#PGIACC_STATIC_O = PgiAcc.o #ENDIF#


#############################################
# Python Binding OBJECTS
#############################################
#PYTHON#PYTHON_O  = PyGroups.o PyExceptions.o PyDatabase.o PyBindings.o PyTimer.o PyTau.o PyMemory.o ctau_impl.o #ENDIF#


OBJS	      = Profiler.o UserEvent.o FunctionInfo.o \
		RtsLayer.o RtsThread.o TauCAPI.o TauFAPI.o TauMapping.o \
		TauHooks.o TauHandler.o TauMemory.o MetaData.o TauEnv.o \
		TauInit.o Comp_intel9.o Comp_pgi.o Comp_xl.o TauCallPath.o \
		$(PARAM_O) Tracer.o TauMetrics.o TauReadMetrics.o $(THR_O) \
		$(PAPI_O) $(TAU_JAVA_O) $(OPENMP_O) $(OPARI_O) \
		$(MULT_O) $(PLATFORM_O) $(PYTHON_O) TauSampling.o \
		TauCompensate.o $(KTAU_O) $(KTAU_MERGE_O) $(KTAU_SHCTR_O) \
		$(RENCI_STFF_O) $(TAU_GASP_O) $(TAU_GASPU_O)

STATIC_OBJS   = Comp_gnu.o $(PGIACC_STATIC_O)
SHARED_OBJS   = $(PGIACC_O)
#SICORTEX#SHARED_OBJS = Comp_gnu.o #ENDIF#
#TAU_BFDSHAREDLINK#SHARED_OBJS = Comp_gnu.o #ENDIF#
#TAU_BFDSHAREDLINK#SHARED_LIBS = $(BFDLINK) -lbfd -liberty #ENDIF#

PRINT	      = pr

SRCS	      = Profiler.cpp UserEvent.cpp FunctionInfo.cpp \
		TauCAPI.cpp TauFAPI.cpp TauMapping.cpp TauHooks.cpp 

AR	      = $(TAU_AR)

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

TAU_TARGET	      = libtau$(TAU_CONFIG)$(LSX)

TAU_DISABLE   = libTauDisable$(LSX)

TAU_DISABLE_SHARED   = libTauDisable$(TAU_SHLIBX)

##############################################
#MPI#MPI_WRAPPER   = libTauMpi$(TAU_CONFIG)$(LSX)  #ENDIF#
#MPI#TAU_MPI_OBJS = TauMpi.o ClockSync.o ProfileMerge.o $(FWRAPPER) $(MPI2EXTENSIONS)  #ENDIF#
#MPI#MPI_INSTALL   = $(INSTALLDEST)/libTauMpi$(TAU_CONFIG)$(LSX)  #ENDIF#
##############################################

##############################################
#SHMEM#SHMEM_WRAPPER   = libTauShmem$(TAU_CONFIG)$(LSX)  #ENDIF#
#SHMEM#SHMEM_INSTALL   = $(INSTALLDEST)/libTauShmem$(TAU_CONFIG)$(LSX)  #ENDIF#
##############################################


##############################################
# For building TauLinuxTimers with gcc
##############################################
#IBM64LINUX#GCC_ABI	     = -m64 		  #ENDIF#



##############################################
# For making libTAU.so Dynamic Shared Object
##############################################
TAU_SHLIBX		= .so 		  
#HP#TAU_SHLIBX		= .sl			#ENDIF#
#APPLECXX#TAU_APPLE_SHLIBX  = .dylib		#ENDIF#
#PYTHON#TAU_APPLE_SHLIBX    = .so		#ENDIF#
#APPLECXX#TAU_SHLIBX    = $(TAU_APPLE_SHLIBX)	#ENDIF#
#GNU#AR_SHFLAGS 		= -shared  	#ENDIF#
#USE_PATHCC#AR_SHFLAGS 		= -shared -fPIC  	#ENDIF#
#PGI#AR_SHFLAGS 		= -shared -fPIC 	#ENDIF#
#APPLEPGI#AR_SHFLAGS 		= -shared #ENDIF#
#SGICC#AR_SHFLAGS 		= -shared  	#ENDIF#
#OPEN64ORC#AR_SHFLAGS 		= -shared -fpic 	#ENDIF#
#APPLECXX#TAU_APPLE_SHFLAGS = -dynamiclib -flat_namespace -undefined suppress #ENDIF#
#PYTHON#TAU_APPLE_SHFLAGS   = -bundle -flat_namespace -undefined suppress     #ENDIF#
#USE_INTELCXX#AR_SHFLAGS  	= -shared 		#ENDIF#
#APPLECXX#AR_SHFLAGS = $(TAU_APPLE_SHFLAGS)     #ENDIF#
#SOL2#AR_SHFLAGS  		= -G 		#ENDIF#
#SUNCC#AR_SHFLAGS  		= -G 		#ENDIF#
#CRAYCNL#AR_SHFLAGS  		= -shared 		#ENDIF#
TAU_BGP_AR_SHFLAGS = -shared
#USE_IBMXLC#AR_SHFLAGS          = -G 		#ENDIF#
#USE_IBMXLC#TAU_BGP_AR_SHFLAGS          = -qmkshrobj                #ENDIF#
#BGP#AR_SHFLAGS          = $(TAU_BGP_AR_SHFLAGS)                #ENDIF#
#USE_DECCXX#AR_SHFLAGS  	= -shared 		#ENDIF#
#ACC#AR_SHFLAGS  		= -b 		#ENDIF#
TAU_ARFLAGS			= $(AR_SHFLAGS) -o 
TAU_SHLIB	= libTAUsh$(TAU_CONFIG)$(TAU_SHLIBX)
#APPLECXX#TAU_SHLIB = $(INSTALLDEST)/libTAUsh$(TAU_CONFIG)$(TAU_SHLIBX) #ENDIF#
#APPLEPGI#TAU_SHLIB = #ENDIF#
#APPLEPGI#TAU_DISABLE_SHARED	= #ENDIF#
#HITACHI#TAU_SHLIB	= #ENDIF#
#HITACHI#TAU_DISABLE_SHARED	= #ENDIF#
#CRAYX1CC#TAU_SHLIB	= #ENDIF#
#CRAYX1CC#TAU_DISABLE_SHARED	= #ENDIF#
#CRAYCC#TAU_SHLIB = #ENDIF#
#CRAYCC#TAU_DISABLE_SHARED	= #ENDIF#
#SUNCC#TAU_SHLIB = #ENDIF#
#SUNCC#TAU_DISABLE_SHARED	= #ENDIF#
#CRAYCC#TAU_NULL=X#ENDIF#
#CRAYX1CC#TAU_NULL=X#ENDIF#
#USE_NECCXX#TAU_SHLIB	= #ENDIF#
#USE_NECCXX#TAU_DISABLE_SHARED	= #ENDIF#
#OPEN64ORC#TAU_SHLIB	= #ENDIF#
#OPEN64ORC#TAU_DISABLE_SHARED	= #ENDIF#
#NOSHARED#TAU_SHLIB	= #ENDIF#
#NOSHARED#TAU_DISABLE_SHARED	= #ENDIF#
#BGL#TAU_SHLIB	= #ENDIF#
#BGL#TAU_DISABLE_SHARED = #ENDIF#
#SOL2CC#EXTRA_LIBS   = -lCstd -liostream -lCrun  #ENDIF#
#SUNCC#EXTRA_LIBS   = -lCstd -liostream -lCrun  #ENDIF#
#CATAMOUNT#TAU_SHLIB = #ENDIF#
#CATAMOUNT#TAU_DISABLE_SHARED = #ENDIF#

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
#PYTHON#TAUPYTHON = pytau$(TAU_SHLIBX) #ENDIF#
#PYTHON#TAUPYTHON_INSTALL = $(TAU_SHAREDDIR)/pytau$(TAU_SHLIBX) #ENDIF#
##############################################
TAU_DSO         = $(TAU_SHLIB)
TAU_SHLIB_INSTALL = $(INSTALLDEST)/$(TAU_SHLIB)
#APPLECXX#TAU_SHLIB_INSTALL= $(TAU_DSO) #ENDIF#

BITS = 32
#IBM64#BITS = 64#ENDIF#

#IBMMPI#POELIB = libmpi_r.a #ENDIF#
#IBMMPI#POELIB_INSTALL = $(INSTALLDEST)/libmpi_r.a #ENDIF#

TAU_SHAREDDIR = $(INSTALLDEST)/shared$(TAU_CONFIG)
TAU_DISABLE_SHAREDDIR = $(INSTALLDEST)/shared-disable

# MPI Wrappers
#MPI#TAU_FORCE_MPI_OBJECT=TauForceMPI.o #ENDIF#

# TAU MPI Wrapper
TAU_MPI_WRAPPER=$(TAU_MPI_OBJS)
TAU_MPI_WRAPPER_FORCE_SHARED=$(TAU_MPI_OBJS) $(TAU_MPI_FLIB)

#SICORTEX#TAU_MPI_FLIB = #ENDIF#

# Epilog MPI Wrapper
#EPILOG#TAU_MPI_WRAPPER=#ENDIF#
#EPILOG#TAU_MPI_WRAPPER_FORCE_SHARED=$(TAU_FORCE_MPI_OBJECT) $(TAU_MPI_FLIB)#ENDIF#
#EPILOG#TAU_FORCE_MPI=$(TAU_FORCE_MPI_OBJECT)#ENDIF#

# VampirTrace MPI Wrapper
#VAMPIRTRACE#TAU_MPI_WRAPPER=#ENDIF#
#VAMPIRTRACE#TAU_MPI_WRAPPER_FORCE_SHARED=$(TAU_FORCE_MPI_OBJECT) $(TAU_MPI_FLIB)#ENDIF#
#VAMPIRTRACE#TAU_FORCE_MPI=$(TAU_FORCE_MPI_OBJECT)#ENDIF#

# Only use the force shared wrapper if -forceshared was specified, since -lelg.mpi
# is often not compiled with -fPIC and cannot be placed in a .so
TAU_MPI_FOR_SHARED=$(TAU_MPI_WRAPPER)
#FORCESHARED#TAU_MPI_FOR_SHARED=$(TAU_MPI_WRAPPER_FORCE_SHARED)#ENDIF#

#SICORTEX#####TAU_MPI_FOR_SHARED= $(TAU_FORCE_MPI_OBJECT) #ENDIF#

CRAY_PGI_HACK=
#CRAYCNL#CRAY_PGI_HACK=$(INSTALLDEST)/libpgnod_prof.a #ENDIF#


all:		$(TAU_TARGET) $(TAU_DISABLE) $(MPI_WRAPPER) $(TAU_DSO) $(TAUJAPI) $(TAUPYTHON) $(SHMEM_WRAPPER) $(POELIB)

$(TAU_TARGET):	$(OBJS) $(STATIC_OBJS)
		$(PRELINK_PHASE)
		$(AR) $(ARFLAGS) $(TAU_TARGET) $(OBJS) $(STATIC_OBJS) $(EXTRAOBJS)
		$(TAU_RANLIB) $(TAU_TARGET)


$(TAU_NULL)$(TAU_SHLIB): 	$(TAU_TARGET) $(OBJS) $(MPI_WRAPPER) $(TAU_FORCE_MPI) $(SHARED_OBJS)
		$(PRELINK_PHASE)
		$(TAU_CXX) $(TAU_ARFLAGS) $(TAU_SHLIB) $(OBJS) $(SHARED_OBJS) $(SHARED_LIBS) $(EXTRAOBJS) $(TAU_MPI_FOR_SHARED) $(EXTRA_LIBS) $(LEXTRA1) $(TAU_PAPI_EXTRA_FLAGS) $(TAU_EXTRA_LIBRARY_FLAGS) $(TAUEXTRASHLIBOPTS)

$(TAU_DISABLE): TauDisable.o $(TAU_DISABLE_SHARED) TauKojakOpariDisable.o
		$(AR) $(ARFLAGS) $(TAU_DISABLE) TauDisable.o TauKojakOpariDisable.o
		$(TAU_RANLIB) $(TAU_DISABLE)

$(TAU_DISABLE_SHARED) : TauDisable.o TauKojakOpariDisable.o
		$(TAU_CXX) $(TAU_ARFLAGS) $(TAU_DISABLE_SHARED) TauDisable.o TauKojakOpariDisable.o

libTauMpi$(TAU_CONFIG)$(LSX): TauMpi.c TauFMpi.c ClockSync.c ProfileMerge.c TauMpiExtensions.c
		$(MAKE) -f Makefile.wrapper  

libTauShmem$(TAU_CONFIG)$(LSX): 
		$(MAKE) -f Makefile.shmem  

$(INSTALLDEST)/$(TAU_TARGET): $(TAU_TARGET)
		@echo Installing $? in $(INSTALLDEST)
		@if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)
		$(TAU_INSTALL) $(TAUROOT)/include/Makefile $(INSTALLDEST)/Makefile.tau$(TAU_CONFIG)

$(INSTALLDEST)/$(TAU_DISABLE): $(TAU_DISABLE) $(TAU_DISABLE_SHARED)
		@echo Installing $? in $(INSTALLDEST)
		@if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)
		cd $(INSTALLDEST)
		@if [ -d $(TAU_DISABLE_SHAREDDIR) ] ; then true; \
		    else mkdir $(TAU_DISABLE_SHAREDDIR); fi
		cd $(TAU_DISABLE_SHAREDDIR) ; rm -f libTAU$(TAU_SHLIBX)
		cd $(TAU_DISABLE_SHAREDDIR) ; ln -s ../$(TAU_DISABLE_SHARED) libTAU$(TAU_SHLIBX)

$(INSTALLDEST)/$(TAU_SHLIB): $(TAU_SHLIB)
		@echo Installing $? in $(INSTALLDEST)
		@if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)
		$(RM) $(INSTALLDEST)/libTAU$(TAU_SHLIBX)
		cd $(INSTALLDEST) && ln -s $(TAU_SHLIB) libTAU$(TAU_SHLIBX)
		@if [ -d $(TAU_SHAREDDIR) ] ; then true; \
		    else mkdir $(TAU_SHAREDDIR); fi
		cd $(TAU_SHAREDDIR) ; rm -f libTAU$(TAU_SHLIBX)
		cd $(TAU_SHAREDDIR) ; ln -s ../$(TAU_SHLIB) libTAU$(TAU_SHLIBX)
		ls $(TAU_SHAREDDIR)

$(INSTALLDEST)/libTauMpi$(TAU_CONFIG)$(LSX): libTauMpi$(TAU_CONFIG)$(LSX)
		$(MAKE) -f Makefile.wrapper install 

$(INSTALLDEST)/libTauShmem$(TAU_CONFIG)$(LSX): libTauShmem$(TAU_CONFIG)$(LSX)
		$(MAKE) -f Makefile.shmem install 

$(TAU_JAVA_INSTALLDIR)/Profile.class: Profile.class
		@echo Installing $? in $(TAU_JAVA_INSTALLDIR)
		@if [ -d $(TAU_JAVA_INSTALLDIR) ] ; then true; \
                   else mkdir -p $(TAU_JAVA_INSTALLDIR) ;fi
		$(TAU_INSTALL) $? $(TAU_JAVA_INSTALLDIR)

$(TAU_SHAREDDIR)/pytau$(TAU_SHLIBX): $(TAU_SHLIB)
		@echo Installing $(TAUPYTHON), ltau.py, and tau.py in $(INSTALLDEST)
		@if [ -d $(INSTALLDEST) ] ; then true; \
		   else mkdir $(INSTALLDEST) ;fi
		$(RM) $(INSTALLDEST)/$(TAUPYTHON) $(INSTALLDEST)/ctau_impl$(TAU_SHLIBX)
		$(TAU_INSTALL) tau.py $(INSTALLDEST)
		$(TAU_INSTALL) ltau.py $(INSTALLDEST)
		cd $(INSTALLDEST) && ln -s $(TAU_SHLIB) $(TAUPYTHON)
		cd $(INSTALLDEST) && ln -s $(TAU_SHLIB) ctau_impl$(TAU_SHLIBX)
		@echo Installing $(TAUPYTHON), ltau.py, and tau.py in $(TAU_SHAREDDIR)
		@if [ -d $(TAU_SHAREDDIR) ] ; then true; \
		    else mkdir $(TAU_SHAREDDIR); fi
		$(RM) $(TAU_SHAREDDIR)/*
		cd $(TAU_SHAREDDIR) && ln -s ../$(TAU_SHLIB) $(TAUPYTHON) && ln -s ../$(TAU_SHLIB) ctau_impl$(TAU_SHLIBX)
		cd $(TAU_SHAREDDIR) && ln -s ../tau.py . && ln -s ../ltau.py .
		cd $(TAU_SHAREDDIR) ; rm -f libTAU$(TAU_SHLIBX)
		cd $(TAU_SHAREDDIR) && ln -s ../$(TAU_SHLIB) libTAU$(TAU_SHLIBX)
		cd $(INSTALLDEST) && ln -s shared$(TAU_CONFIG) bindings$(TAU_CONFIG)

$(INSTALLDEST)/libpgnod_prof.a : 
	ar rcs $(INSTALLDEST)/libpgnod_prof.a


# break apart libelg.mpi.a to build into libmpi_r.a for tau_poe (AIX)
epilog_breakup: 
	@echo Breaking up libelg.mpi.a
	$(RM) $(BITS)
	mkdir -p $(BITS)/tau
	cd $(BITS)/tau ; ar -X $(BITS) xv $(EPILOGLIBDIR)/libelg.mpi.a

# place TAU objects into libmpi_r.a for tau_poe (AIX)
tau_breakup: $(TAU_SHLIB) $(MPI_WRAPPER)
	$(RM) $(BITS)
	mkdir -p $(BITS)/tau
	cp $(TAU_TARGET) libTauMpi$(TAU_CONFIG)$(LSX) $(BITS)/tau

# break apart vampirtrace to build into libmpi_r.a for tau_poe (AIX)
vampirtrace_breakup:
	@echo Breaking up VampirTrace
	$(RM) $(BITS)
	mkdir -p $(BITS)/tau
	cd $(BITS)/tau ; for i in $(VAMPIRTRACEDIR)/lib/*.a ; do ar -X $(BITS) xv $$i ; done


MPI_BREAKUP_PREREQ = tau_breakup
#EPILOG#MPI_BREAKUP_PREREQ = epilog_breakup #ENDIF#
#VAMPIRTRACE#MPI_BREAKUP_PREREQ = vampirtrace_breakup #ENDIF#


libmpi_r.a: $(MPI_BREAKUP_PREREQ)
	echo Creating libmpi_r.a
	./rebuild_libmpi $(BITS) $(TAU_LINKER_OPTS)
	cp $(BITS)/libmpi_r.a .
	$(RM) $(BITS)

$(INSTALLDEST)/libmpi_r.a: libmpi_r.a
	echo Installing libmpi_r.a in $(TAU_SHAREDDIR)
	if [ -d $(INSTALLDEST) ] ; then true; \
	   else mkdir $(INSTALLDEST) ;fi
	if [ -d $(TAU_SHAREDDIR) ] ; then true; \
	    else mkdir $(TAU_SHAREDDIR); fi
	$(RM) $(TAU_SHAREDDIR)/libmpi_r.a
	$(RM) -f $(TAU_SHAREDDIR)/libmpi_r.a
	$(RM) -f $(INSTALLDEST)/libmpi_r.a
	cp libmpi_r.a $(TAU_SHAREDDIR)
	$(RM) -f $(INSTALLDEST)/bindings-default
	cd $(INSTALLDEST) && ln -s shared$(TAU_CONFIG) bindings$(TAU_CONFIG)
	cd $(INSTALLDEST) && ln -s bindings$(TAU_CONFIG) bindings-default

install:	 $(INSTALLDEST)/$(TAU_TARGET) $(INSTALLDEST)/$(TAU_DISABLE) $(MPI_INSTALL) $(TAU_SHLIB_INSTALL) $(TAU_JAVA_CLASS1) $(TAUPYTHON_INSTALL) $(SHMEM_INSTALL) $(POELIB_INSTALL) $(CRAY_PGI_HACK)

TauForceMPI.o : TauForceMPI.c
	$(CC) $(CFLAGS) -c $(@:.o=.c)

# Rules
Profile.class	: Profile.java
	$(JDKDIR)/bin/javac Profile.java 

Profiler.o 	: Profiler.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

MetaData.o 	: MetaData.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauSampling.o 	: TauSampling.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauEnv.o 	: TauEnv.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauInit.o 	: TauInit.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

UserEvent.o 	: UserEvent.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

FunctionInfo.o 	: FunctionInfo.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

Comp_gnu.o 	: Comp_gnu.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $(BFDINCLUDE) -c $(@:.o=.cpp)

Comp_intel9.o 	: Comp_intel9.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

Comp_pgi.o 	: Comp_pgi.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

Comp_xl.o 	: Comp_xl.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

PgiAcc.o 	: PgiAcc.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

RtsLayer.o 	: RtsLayer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

RtsThread.o 	: RtsThread.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

ProfileParam.o   : ProfileParam.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

Tracer.o   	: Tracer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauMetrics.o   	: TauMetrics.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauReadMetrics.o : TauReadMetrics.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

PthreadLayer.o   : PthreadLayer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

SprocLayer.o   : SprocLayer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

PapiThreadLayer.o   : PapiThreadLayer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

JavaThreadLayer.o   : JavaThreadLayer.cpp $(HDRS)
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

TauKojakOpari.o : TauKojakOpari.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauKojakOpariDisable.o : TauKojakOpariDisable.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauLinuxTimers.o : TauLinuxTimers.c $(HDRS)
	gcc -c -fPIC $(GCC_ABI) $(USER_OPT) $(@:.o=.c)

TauCallPath.o 	: TauCallPath.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

PyGroups.o 	: PyGroups.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

PyExceptions.o 	: PyExceptions.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

PyDatabase.o 	: PyDatabase.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

PyBindings.o 	: PyBindings.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

PyTimer.o 	: PyTimer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

PyTau.o 	: PyTau.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

ctau_impl.o 	: ctau_impl.c $(HDRS)
	$(CC) $(CFLAGS) -c $(@:.o=.c)

TauCompensate.o : TauCompensate.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauHandler.o : TauHandler.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauMemory.o : TauMemory.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TauGASP.o : TauGASP.c $(HDRS)
	$(TAU_UPCC) $(UPCFLAGS) -c $(@:.o=.c)

TauGASPU.o : TauGASPU.upc $(HDRS)
	$(TAU_UPCC) $(UPCFLAGS) -c $(@:.o=.upc)


ktau_syscall.o : ktau_syscall.c $(HDRS)
	$(CC) $(CXXFLAGS) -c $(@:.o=.c)

$(TAUROOT)/src/Profile/TauPGIHelper.o 	: TauPGIHelper.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp) -o $@


clean : 
	$(RM) *.ii *.ti ti_files cxx_repository ii_files rii_files *.a *.o *$(TAU_SHLIBX)

