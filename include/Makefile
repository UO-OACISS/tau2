#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/tau	           **
#****************************************************************************
#*    Copyright 1997-2002        				   	   **
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
TAU_ARCH=default
CONFIG_CC=gcc
CONFIG_CXX=g++
TAU_CC_FE=$(CONFIG_CC)
#BGL#TAU_CC_FE=xlc #ENDIF#
#CATAMOUNT#TAU_CC_FE=gcc #ENDIF#
PCXX_OPT=-g
USER_OPT=-g
EXTRADIR=
EXTRADIRCXX=
TAUROOT=
TULIPDIR=
TAUGCCLIBOPTS=
TAUGCCLIBDIR=
PCLDIR=
PAPIDIR=
PAPISUBDIR=
CHARMDIR=
PDTDIR=
PDTCOMPDIR=
DYNINSTDIR=
JDKDIR=
SLOG2DIR=
OPARIDIR=
TAU_OPARI_TOOL=
EPILOGDIR=
KTAU_INCDIR=
KTAU_KALLSYMS_PATH=
PYTHON_INCDIR=
PYTHON_LIBDIR=
TAU_SHMEM_INC=
TAU_SHMEM_LIB=
TAU_CONFIG=
TAU_MPI_INC=
TAU_MPI_LIB=
TAU_MPI_FLIB=
TAU_MPILIB_DIR=
TAU_PREFIX_INSTALL_DIR=
#######################################################################
 
#ENABLE64BIT#ABI 	     = -64		  #ENDIF#
#ENABLEN32BIT#ABI 	     = -n32		  #ENDIF#
#ENABLE32BIT#ABI 	     = -32		  #ENDIF#

#######################################################################
#IBM64#IBM_XLC_ABI	     = -q64 		  #ENDIF#
#IBM64#IBM_GNU_ABI	     = -maix64 		  #ENDIF#
#IBM64LINUX#IBM_XLC_ABI	     = -q64 		  #ENDIF#
#IBM64LINUX#IBM_GNU_ABI	     = -m64 		  #ENDIF#
#SUNX86_64#SUN_GNU_ABI       = -m64		  #ENDIF#
#SUNX86_64#SUN_CC_ABI        = -xarch=amd64       #ENDIF#

#USE_IBMXLC#IBM_ABI    	     = $(IBM_XLC_ABI) 	  #ENDIF#
#GNU#IBM_ABI	     	     = $(IBM_GNU_ABI) 	  #ENDIF#
#SP1# ABI		     = $(IBM_ABI)	  #ENDIF#
#PPC64# ABI		     = $(IBM_ABI)	  #ENDIF#
#SOLARIS64#SUN_GNU_ABI	     = -mcpu=v9 -m64	  #ENDIF#
#SOLARIS64#SUN_CC_ABI	     = -xarch=v9 -xcode=pic32	  #ENDIF#
#SOL2CC#SUN_ABI	     	     = $(SUN_CC_ABI)	  #ENDIF#
#GNU#SUN_ABI	     	     = $(SUN_GNU_ABI)	  #ENDIF#
#SOL2#ABI	     	     = $(SUN_ABI)	  #ENDIF#
#SUNX86_64#ABI		     = $(SUN_ABI)	  #ENDIF#
#######################################################################
F90_ABI	= $(ABI) 
#IBM64_FORTRAN#F90_ABI	     = -q64 		  #ENDIF#
#######################################################################

############# Standard Defines ##############
TAU_CC = $(CONFIG_CC) $(ABI) $(ISA)
TAU_CXX = $(CONFIG_CXX) $(ABI) $(ISA) $(USER_OPT)
TAU_INSTALL = /bin/cp
TAU_SHELL = /bin/sh
LSX = .a
#############################################
# JAVA DEFAULT ARCH 
#############################################
JDKARCH                 = linux
#COMPAQ_ALPHA#JDKARCH   = alpha      #ENDIF#
#SOL2#JDKARCH           = solaris    #ENDIF#
#SGIMP#JDKARCH          = irix       #ENDIF#
#SP1#JDKARCH            = aix        #ENDIF#
#T3E#JDKARCH            = cray       #ENDIF#
#############################################
# JAVA OBJECTS
#############################################
#JAVA#TAU_JAVA_O 	= TauJava.o TauJAPI.o  #ENDIF#
#JAVA#TAUJAPI 	= Profile.class        #ENDIF#


#############################################
# OpenMP OBJECTS
#############################################
#OPENMP#OPENMP_O 	= OpenMPLayer.o #ENDIF#
#GNU#OPENMP_O 	= #ENDIF#
# GNU G++ does not currently support OpenMP 

#############################################
# Opari OBJECTS
#############################################
#OPARI#OPARI_O 	= TauOpari.o #ENDIF#
#KOJAKOPARI#OPARI_O 	= TauKojakOpari.o #ENDIF#
#EPILOG#OPARI_O 	=  #ENDIF#
#GNU#OPARI_O 	= #ENDIF#

#############################################
# CallPath OBJECTS
#############################################
#PROFILECALLPATH#CALLPATH_O  = TauCallPath.o #ENDIF#
#PROFILEPARAM#PARAM_O  = ProfileParam.o #ENDIF#

#############################################
# Python Binding OBJECTS
#############################################
#PYTHON#PYTHON_O  = PyGroups.o PyExceptions.o PyDatabase.o PyBindings.o PyTimer.o PyTau.o #ENDIF#

#############################################
# DYNINST DEFAULT ARCH
#############################################
DYNINST_PLATFORM 	= $(PLATFORM)


#PCL##include $(TAU_PREFIX_INSTALL_DIR)/include/makefiles/PCLMakefile.stub	  #ENDIF#

############# OpenMP Fortran Option ########
#OPENMP#TAU_F90_OPT = -mp                   #ENDIF#
#SOL2CC_OPENMP#TAU_F90_OPT = -xopenmp       #ENDIF#
#COMPAQCXX_OPENMP#TAU_F90_OPT = -omp        #ENDIF#
#IBMXLC_OPENMP#TAU_F90_OPT = -qsmp=omp      #ENDIF#
#GUIDE#TAU_F90_OPT =                        #ENDIF#
#PGIOPENMP#TAU_F90_OPT = -mp                #ENDIF#
#INTELOPENMP#TAU_F90_OPT = -openmp          #ENDIF#
#HITACHI_OPENMP#TAU_F90_OPT =               #ENDIF#

#THREADSAFE_COMPILERS#TAU_R         =_r     #ENDIF#

############# Fortran Compiler #############
#GNU_FORTRAN#TAU_F90         = g77              $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#GNU_GFORTRAN#TAU_F90         = gfortran              $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#SGI_FORTRAN#TAU_F90         = f90              $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#IBM_FORTRAN#TAU_F90         = xlf90$(TAU_R)    $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#BGL#TAU_F90         = blrts_xlf90$(TAU_R)    $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#IBM64_FORTRAN#TAU_F90       = xlf90$(TAU_R)    $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#IBMXLFAPPLE#TAU_F90       = xlf90$(TAU_R)    $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#CRAY_FORTRAN#TAU_F90        = f90              $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#CRAY_X1_FORTRAN#TAU_F90     = ftn              $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#PGI_FORTRAN#TAU_F90         = pgf90            $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#PGI_CATAMOUNT#TAU_F90         = qk-pgf90         $(F90_ABI) $(TAU_F90_OPT) #ENDIF#
#ABSOFT_FORTRAN#TAU_F90      = f90              $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#LAHEY_FORTRAN#TAU_F90      = lf95              $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#NAGWARE_FORTRAN#TAU_F90      = f95              $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#FUJITSU_FORTRAN#TAU_F90     = F90              $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#FUJITSU_SOLARIS#TAU_F90     = f90              $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#SUN_FORTRAN#TAU_F90         = f90              $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#COMPAQ_FORTRAN#TAU_F90      = f90              $(F90_ABI)  $(TAU_F90_OPT)  #ENDIF#
#KAI_FORTRAN#TAU_F90         = guidef90         $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#HP_FORTRAN#TAU_F90          = f90 	        $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#HITACHI_FORTRAN#TAU_F90     = f90 	        $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#INTEL_FORTRAN#TAU_F90       = efc 	        $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#INTEL32_FORTRAN#TAU_F90     = ifc 	        $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#INTELIFORT#TAU_F90     = ifort 	        $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#PATHSCALE_FORTRAN#TAU_F90     = pathf90         $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#OPEN64ORC_FORTRAN#TAU_F90     = orf90         $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#
#NEC_FORTRAN#TAU_F90         = f90 	        $(F90_ABI) $(TAU_F90_OPT)   #ENDIF#


############# Portable F90 Options #############
#IBM64_FORTRAN#TAU_F90_FIXED      = -qfixed  #ENDIF#
#IBM_FORTRAN#TAU_F90_FIXED      = -qfixed  #ENDIF#
#IBM_FORTRAN#TAU_F90_SUFFIX     = -qsuffix=f=f90  #ENDIF#
#IBMXLFAPPLE#TAU_F90_FIXED      = -qfixed  #ENDIF#
#IBMXLFAPPLE#TAU_F90_SUFFIX     = -qsuffix=f=f90  #ENDIF#
#IBM64_FORTRAN#TAU_F90_SUFFIX     = -qsuffix=f=f90  #ENDIF#

############# Profiling Options #############
#PROFILE#PROFILEOPT1           = -DPROFILING_ON        #ENDIF#
#PCL#PROFILEOPT3       	       = -DTAU_PCL -I$(PCLDIR)/include #ENDIF#
#PAPI#PROFILEOPT3              = -DTAU_PAPI -I$(PAPIDIR)/src -I$(PAPIDIR)/include #ENDIF#
#PCL#PCL_O       	       = PclLayer.o 	       #ENDIF#
#PAPI#PAPI_O       	       = PapiLayer.o 	       #ENDIF#
#MULTIPLECOUNTERS#MULT_O       = MultipleCounters.o    #ENDIF#
#PROFILECALLS#PROFILEOPT4      = -DPROFILE_CALLS       #ENDIF#
#PROFILESTATS#PROFILEOPT5      = -DPROFILE_STATS       #ENDIF#
#DEBUGPROF#PROFILEOPT6         = -DDEBUG_PROF          #ENDIF#
#STDCXXLIB#PROFILEOPT7         = -DTAU_STDCXXLIB       #ENDIF#
#CRAYX1CC#PROFILEOPT7         = #ENDIF#
#CRAYCC#PROFILEOPT7         = #ENDIF#
#INTELTFLOP#PROFILEOPT8        = -DPOOMA_TFLOP         #ENDIF#
#NORTTI#PROFILEOPT9            = -DNO_RTTI             #ENDIF#
#RTTI#PROFILEOPT9              = -DRTTI             #ENDIF#
#GNU#PROFILEOPT10              = -DTAU_GNU  -DTAU_DOT_H_LESS_HEADERS -fPIC #ENDIF#
#APPLECXX#PROFILEOPT10              = -DTAU_GNU  -DTAU_DOT_H_LESS_HEADERS -fPIC #ENDIF#
#USE_PATHCC#PROFILEOPT10       = -DTAU_DOT_H_LESS_HEADERS -fPIC -DTAU_PATHSCALE #ENDIF#
#OPEN64ORC#PROFILEOPT10       = -DTAU_DOT_H_LESS_HEADERS -DTAU_OPEN64ORC -fpic  #ENDIF#
#CALLSTACK#PROFILEOPT11        = -DPROFILE_CALLSTACK   #ENDIF#
#PGI1.7#PROFILEOPT12	       = -DPGI 		       #ENDIF#
#CRAYKAI#PROFILEOPT12	       = -DCRAYKAI	       #ENDIF#
#HP_FORTRAN#PROFILEOPT12       = -DHP_FORTRAN	       #ENDIF#
#CRAYCC#PROFILEOPT13	       = -h instantiate=used -DCRAYCC -DTAU_DOT_H_LESS_HEADERS  #ENDIF#
#CRAYX1CC#PROFILEOPT13	       = -DTAU_DOT_H_LESS_HEADERS #ENDIF#
#SGICC#TAU_CXX = $(CONFIG_CXX) $(ABI) $(ISA) $(USER_OPT) -LANG:std  #ENDIF#
#PTHREAD_AVAILABLE#PROFILEOPT15 = -DPTHREADS  	       #ENDIF#
#COMPAQCXX_PTHREAD#PROFILEOPT15 = -DPTHREADS -pthread  #ENDIF#
#TAU_SPROC#PROFILEOPT15 	= -DTAU_SPROC  	       #ENDIF#
#TULIPTHREADS#PROFILEOPT16 = -DTULIPTHREADS            #ENDIF#
#TRACE#TRACEOPT		       = -DTRACING_ON	       #ENDIF#
#TRACE#EVENTS_O                = Tracer.o              #ENDIF#
#KTAU#KTAU_O			= ktau_proc_interface.o TauKtau.o KtauProfiler.o KtauSymbols.o   #ENDIF#
#KTAU_MERGE#KTAU_MERGE_O      	= KtauFuncInfo.o KtauMergeInfo.o   #ENDIF#
#MPITRACE#TRACEOPT	 = -DTAU_MPITRACE -DTRACING_ON #ENDIF#
#MPITRACE#EVENTS_O                = Tracer.o              #ENDIF#
#MUSE#MUSE_O                  = TauMuse.o TauMuseFilters.o TauMuseHandlers.o TauMusePackages.o #ENDIF#
#MUSE_EVENT#MUSE_O                  = TauMuse.o TauMuseFilters.o TauMuseHandlers.o TauMusePackages.o #ENDIF#
#MUSE_MULTIPLE#MUSE_O                  = TauMuse.o TauMuseFilters.o TauMuseHandlers.o TauMusePackages.o #ENDIF#
#COMPENSATE#COMPENSATE_O      = TauCompensate.o #ENDIF#
#PTHREAD_AVAILABLE#THR_O       = PthreadLayer.o        #ENDIF#
#TAU_SPROC#THR_O       	       = SprocLayer.o        #ENDIF#
#JAVA#THR_O       	       = JavaThreadLayer.o     #ENDIF#
#TULIPTHREADS#THR_O       = TulipThreadLayer.o         #ENDIF#
#LINUXTIMERS#PLATFORM_O       = TauLinuxTimers.o       #ENDIF#
#TULIPTHREADS#PROFILEOPT17  = -I$(TULIPDIR)/include -I$(TULIPDIR)/Tuliplib   #ENDIF#
#SMARTS#PROFILEOPT17  = -I$(TULIPDIR)/include -I$(TULIPDIR)/lib -I$(TULIPDIR)/machine-specific/$(HOSTTYPE)  #ENDIF#
#SMARTS#PROFILEOPT18  = -DSMARTS   #ENDIF#
#KAI#PROFILEOPT19  	       = -DKAI  -DTAU_DOT_H_LESS_HEADERS #ENDIF#
#USE_DECCXX#PROFILEOPT19  	       = -DTAU_DOT_H_LESS_HEADERS   #ENDIF#
#USE_INTELCXX#PROFILEOPT19     = -DTAU_DOT_H_LESS_HEADERS   -fPIC #ENDIF#
#USE_NECCXX#PROFILEOPT19     = -DTAU_DOT_H_LESS_HEADERS   #ENDIF#
#PGI#PROFILEOPT19  	       = -DTAU_DOT_H_LESS_HEADERS   -fPIC #ENDIF#
#ACC#PROFILEOPT19  	       = -AA +z -DTAU_DOT_H_LESS_HEADERS -DTAU_HPUX #ENDIF#
#FUJITSU#PROFILEOPT19  	       = -DFUJITSU -DTAU_DOT_H_LESS_HEADERS   #ENDIF#
#KAINOEX#PROFILEOPT20  	       = --no_exceptions   #ENDIF#
#SGICCNOEX#PROFILEOPT20  	       = -LANG:exceptions=off  #ENDIF#
#HPGNU#PROFILEOPT21	       = -fPIC #ENDIF#
#HITACHI#PROFILEOPT21	       = -DTAU_HITACHI #ENDIF#
#SP1#PROFILEOPT21	       = -D_POSIX_SOURCE -DTAU_AIX #ENDIF#
#PPC64#TAU_PIC_PROFILEOPT21           = -qpic=large #ENDIF#
#BGL#TAU_PIC_PROFILEOPT21           = #ENDIF#
#USE_IBMXLC#PROFILEOPT21	       = -DTAU_DOT_H_LESS_HEADERS -DTAU_XLC $(TAU_PIC_PROFILEOPT21) #ENDIF#
#IBMXLCAPPLE#PROFILEOPT21	       = -DTAU_DOT_H_LESS_HEADERS -DTAU_XLC -DTAU_APPLE_XLC #ENDIF#
#PCLPTHREAD#PROFILEOPT22	       = -DPCL_MUTEX_LOCK #ENDIF#
#JAVA#PROFILEOPT23	       = -DJAVA 		#ENDIF#
#MONITOR#PROFILEOPT24	       = -DMONITORING_ON 	#ENDIF#
#JAVA#PROFILEOPT25 = -I$(JDKDIR)/include -I$(JDKDIR)/include/$(JDKARCH) #ENDIF#
#MPI#PROFILEOPT26 = -DTAU_MPI #ENDIF#
#MPI_THREADED#PROFILEOPT26 = -DTAU_MPI -DTAU_MPI_THREADED #ENDIF#
#OPENMP#PROFILEOPT27 = -mp -DTAU_OPENMP#ENDIF#
#GNU#PROFILEOPT27 = #ENDIF#
#SOL2CC_OPENMP#PROFILEOPT27 = -xopenmp -DTAU_OPENMP#ENDIF#
#HITACHI_OPENMP#PROFILEOPT27 = -DTAU_OPENMP#ENDIF#
#COMPAQCXX_OPENMP#PROFILEOPT27 = -omp -DTAU_OPENMP#ENDIF#
#IBMXLC_OPENMP#PROFILEOPT27 = -qsmp=omp -DTAU_OPENMP #ENDIF#
#OPEN64_OPENMP#PROFILEOPT27 = -mp -DTAU_OPENMP #ENDIF#
#GUIDE#PROFILEOPT27 = -DTAU_OPENMP #ENDIF#
#PGIOPENMP#PROFILEOPT27 = -mp -D_OPENMP -DTAU_OPENMP -U_RWSTD_MULTI_THREAD -U_REENTRANT #ENDIF#
#INTELOPENMP#PROFILEOPT27 = -openmp -DTAU_OPENMP #ENDIF#
#OPARI#PROFILEOPT28 = -I$(OPARIDIR)/lib -I$(OPARIDIR)/include #ENDIF#
#OPARI_REGION#PROFILEOPT28 = -DTAU_OPARI_REGION -I$(OPARIDIR)/lib -I$(OPARIDIR)/include #ENDIF#
#OPARI_CONSTRUCT#PROFILEOPT28 = -DTAU_OPARI_CONSTRUCT -I$(OPARIDIR)/lib -I$(OPARIDIR)/include #ENDIF#
#MULTIPLECOUNTERS#PROFILEOPT29 = -DTAU_MULTIPLE_COUNTERS #ENDIF#
#SGITIMERS#PROFILEOPT30         = -DSGI_TIMERS          #ENDIF#
#CRAYTIMERS#PROFILEOPT30         = -DCRAY_TIMERS          #ENDIF#
#LINUXTIMERS#PROFILEOPT31       = -DTAU_LINUX_TIMERS    #ENDIF#
#ALPHATIMERS#PROFILEOPT31       = -DTAU_ALPHA_TIMERS    #ENDIF#
#CPUTIME#PROFILEOPT32           = -DCPU_TIME          #ENDIF#
#PAPIWALLCLOCK#PROFILEOPT33     = -DTAU_PAPI_WALLCLOCKTIME    #ENDIF#
#PAPIVIRTUAL#PROFILEOPT34       = -DTAU_PAPI_VIRTUAL    #ENDIF#
#SGICOUNTERS#PROFILEOPT35      = -DSGI_HW_COUNTERS     #ENDIF#
#EPILOG#PROFILEOPT36	       = -DTAU_EPILOG -I$(EPILOGDIR)/src -I$(EPILOGDIR)/include	#ENDIF#
#PROFILECALLPATH#PROFILEOPT36  = -DTAU_CALLPATH #ENDIF#
#PROFILEPHASE#PROFILEOPT36  = -DTAU_CALLPATH -DTAU_PROFILEPHASE#ENDIF#
#PYTHON#PROFILEOPT37  = -I$(PYTHON_INCDIR) #ENDIF#
#NOCOMM#PROFILEOPT38  = -DTAU_NOCOMM #ENDIF#
#MUSE#PROFILEOPT39  = -DTAU_MUSE #ENDIF#
#SETNODE0#PROFILEOPT40  = -DTAU_SETNODE0 #ENDIF#
#COMPENSATE#PROFILEOPT41  = -DTAU_COMPENSATE #ENDIF#
#MUSE_EVENT#PROFILEOPT42  = -DTAU_MUSE_EVENT #ENDIF#
#MUSE_MULTIPLE#PROFILEOPT43  = -DTAU_MUSE_MULTIPLE #ENDIF#
#DYNINST41##PROFILEOPT44  = -DTAU_DYNINST41BUGFIX #ENDIF#
# DyninstAPI v4.2.1 fixes the bug, so we don't need OPT44 anymore
#PROFILEMEMORY#PROFILEOPT45  = -DTAU_PROFILEMEMORY   #ENDIF#
#MPIGREQUEST#PROFILEOPT46  = -DTAU_MPIGREQUEST   #ENDIF#
#MPIOREQUEST#PROFILEOPT47  = -DTAU_MPIOREQUEST   #ENDIF#
#MPIDATAREP#PROFILEOPT48  = -DTAU_MPIDATAREP   #ENDIF#
#MPIERRHANDLER#PROFILEOPT49  = -DTAU_MPIERRHANDLER  #ENDIF#
#CATAMOUNT#PROFILEOPT50  = -DTAU_CATAMOUNT  #ENDIF#
#MPICONSTCHAR#PROFILEOPT51  = -DTAU_MPICONSTCHAR  #ENDIF#
#MPIATTR#PROFILEOPT52  = -DTAU_MPIATTRFUNCTION   #ENDIF#
#MPITYPEEX#PROFILEOPT53  = -DTAU_MPITYPEEX   #ENDIF#
#MPIADDERROR#PROFILEOPT54  = -DTAU_MPIADDERROR   #ENDIF#
#MPINEEDSTATUSCONV#PROFILEOPT55  = -DTAU_MPI_NEEDS_STATUS   #ENDIF#

#DEPTHLIMIT#PROFILEOPT56      = -DTAU_DEPTH_LIMIT       #ENDIF#
#TAU_CHARM#PROFILEOPT57 = -DTAU_CHARM -I$(CHARMDIR)/include #ENDIF#
#PROFILEHEADROOM#PROFILEOPT58  = -DTAU_PROFILEHEADROOM   #ENDIF#
#JAVACPUTIME#PROFILEOPT59           = -DJAVA_CPU_TIME          #ENDIF#
#TAU_LARGEFILE#PROFILEOPT60           = -DTAU_LARGEFILE -D_LARGEFILE64_SOURCE     #ENDIF#
#SHMEM#PROFILEOPT61           = -DTAU_SHMEM #ENDIF#
#KTAU#PROFILEOPT62  = -DTAUKTAU -DKTAU_USER_SRC_COMPILE -I$(KTAU_INCDIR) -DKTAU_KALLSYMS_PATH=\"$(KTAU_KALLSYMS_PATH)\" #ENDIF#
#KTAU_MERGE#PROFILEOPT63  = -DTAUKTAU_MERGE -DKTAU_USER_SRC_COMPILE -I$(KTAU_INCDIR) -DKTAU_KALLSYMS_PATH=\"$(KTAU_KALLSYMS_PATH)\" #ENDIF#
#FREEBSD#PROFILEOPT64  = -DTAU_FREEBSD #ENDIF#
#PROFILEPARAM#PROFILEOPT65  = -DTAU_PROFILEPARAM #ENDIF#
#IBMMPI#PROFILEOPT66  = -DTAU_IBM_MPI #ENDIF#
#WEAKMPIINIT#PROFILEOPT67  = -DTAU_WEAK_MPI_INIT   #ENDIF#



#For F90 support for all platforms
FWRAPPER	       = TauFMpi.o       
#MPI2#MPI2EXTENSIONS	       = TauMpiExtensions.o       #ENDIF#
#CRAYX1CC#MPI2EXTENSIONS	       =        #ENDIF#

#SGICOUNTERS#LEXTRA	       = -lperfex 	       #ENDIF#
#ALPHATIMERS#LEXTRA	       = -lrt 	       #ENDIF#
#SOL2#PCL_EXTRA_LIBS = -lcpc #ENDIF#
#PCL#LEXTRA	       	       = -L$(PCLDIR)/lib -lpcl $(PCL_EXTRA_LIBS)      #ENDIF#
#PAPI#LEXTRA	       	       = $(PAPIDIR)/$(PAPISUBDIR)/libpapi.a #ENDIF#
#IA64PAPI#LEXTRA	       = -L$(PAPIDIR)/$(PAPISUBDIR) -lpapi #ENDIF#
#SOL2PAPI#LEXTRA	       = $(PAPIDIR)/$(PAPISUBDIR)/libpapi.a -lcpc #ENDIF#
#IBMPAPI#LEXTRA  = $(PAPIDIR)/$(PAPISUBDIR)/libpapi.a -L/usr/lpp/pmtoolkit/lib -L/usr/pmapi/lib -lpmapi#ENDIF#
#PPC64PAPI#LEXTRA  = $(PAPIDIR)/$(PAPISUBDIR)/libpapi.a #ENDIF#
#BGLPAPI#LEXTRA  = $(PAPIDIR)/$(PAPISUBDIR)/libpapi.rts.a -L/bgl/BlueLight/ppcfloor/bglsys/lib -lbgl_perfctr.rts -lrts.rts -ldevices.rts #ENDIF#
#X86_64PAPI#LEXTRA  = -L$(PAPIDIR)/$(PAPISUBDIR)/ -L$(PAPIDIR)/lib64/ -lpapi -lperfctr #ENDIF#
#IBM64PAPI#LEXTRA = $(PAPIDIR)/$(PAPISUBDIR)/libpapi64.a -L/usr/lpp/pmtoolkit/lib  -L/usr/pmapi/lib -lpmapi #ENDIF#
#IBM64PAPILINUX#LEXTRA = $(PAPIDIR)/$(PAPISUBDIR)/libpapi.a #ENDIF#
#SGI64PAPI#LEXTRA  = -L$(PAPIDIR)/$(PAPISUBDIR) -lpapi64 #ENDIF#
#ALPHAPAPI#LEXTRA	       	       = $(PAPIDIR)/$(PAPISUBDIR)/libpapi.a /usr/lib/dcpi/dadd.a -lclu -lrt #ENDIF#

##PAPI#TAU_PAPI_RPATH	= -rpath $(PAPIDIR)/$(PAPISUBDIR) #ENDIF#
#PPC64PAPI#TAU_PAPI_RPATH = #ENDIF#
#BGLPAPI#TAU_PAPI_RPATH = #ENDIF#
#USE_INTELCXX#TAU_PAPI_RPATH	=  #ENDIF#
#CRAYX1CC#TAU_PAPI_RPATH = #ENDIF#
#PGI#TAU_PAPI_RPATH	= -R $(PAPIDIR)/$(PAPISUBDIR) #ENDIF#
#GNU#TAU_PAPI_RPATH	= -Wl,-rpath $(PAPIDIR)/$(PAPISUBDIR) #ENDIF#
#SOL2PAPI#TAU_PAPI_RPATH	= #ENDIF#
#IBMPAPI#TAU_PAPI_RPATH	= #ENDIF#
#IBM64PAPI#TAU_PAPI_RPATH	= #ENDIF#
#PAPI#TAU_LINKER_OPT1 = $(TAU_PAPI_RPATH) #ENDIF#

#PTHREAD_AVAILABLE#LEXTRA1     = -lpthread	       #ENDIF#
#TULIPTHREADS#LEXTRA1     = -L$(TULIPDIR)/Tuliplib  -ltulip	       #ENDIF#
#SMARTS##include $(TAU_PREFIX_INSTALL_DIR)/include/makefiles/GNUmakefile-$(HOSTTYPE)  #ENDIF#
#SMARTS#LEXTRA1	    = $(LSMARTS)	#ENDIF#

TAU_GCCLIB	= -lgcc_s 
#FREEBSD#TAU_GCCLIB	= -lgcc #ENDIF#
#BGL#TAU_GCCLIB	= -lgcc #ENDIF#
#GNU#TAU_FORTRANLIBS       = -L$(TAUGCCLIBDIR) $(TAUGCCLIBOPTS) -lstdc++ $(TAU_GCCLIB) #ENDIF#
#OPEN64ORC_FORTRAN#TAU_FORTRANLIBS       = -lfortran -lffio #ENDIF#
#PATHSCALE_FORTRAN#TAU_FORTRANLIBS       = -lpathfstart -lpathfortran #ENDIF#
#NAGWARE_FORTRAN#TAU_FORTRANLIBS		= $(EXTRADIR)/lib/quickfit.o -L$(EXTRADIR)/lib -lf96 #ENDIF#
#GNU_FORTRAN#TAU_FORTRANLIBS	       = -lg2c       #ENDIF#
#GNU_GFORTRAN#TAU_FORTRANLIBS	       = -lg2c       #ENDIF#
#SGI_FORTRAN#TAU_FORTRANLIBS	       = -lfortran -lftn       #ENDIF#
#USE_IBMXLC#TAU_IBM_FORTRANLIBS	       =  -bh:4 -bpT:0x10000000 -bpD:0x20000000 /lib/crt0.o -lxlf90 -lm -lc #ENDIF#
#GNU#TAU_IBM_FORTRANLIBS	       =  -Wl,-bh:4 -Wl,-bpT:0x10000000 -Wl,-bpD:0x20000000 /lib/crt0.o -lxlf90 -lm -lc #ENDIF#
#KAI#TAU_IBM_FORTRANLIBS	       =  --backend -bh:4 --backend -bpT:0x10000000 --backend -bpD:0x20000000 /lib/crt0.o -lxlf90 -lm -lc #ENDIF#
#IBM_FORTRAN#TAU_FORTRANLIBS = $(TAU_IBM_FORTRANLIBS) #ENDIF#

#USE_IBMXLC#TAU_IBM64_FORTRANLIBS      =  -bh:4 -bpT:0x10000000 -bpD:0x20000000 -b64 /lib/crt0_64.o -lxlf90 -lm -lc #ENDIF#
#GNU#TAU_IBM64_FORTRANLIBS      =  -Wl,-bh:4 -Wl,-bpT:0x10000000 -Wl,-bpD:0x20000000 -Wl,-b64 /lib/crt0_64.o -lxlf90 -lm -lc #ENDIF#
#KAI#TAU_IBM64_FORTRANLIBS      =  --backend -bh:4 --backend -bpT:0x10000000 --backend -bpD:0x20000000 --backend -b64 /lib/crt0_64.o -lxlf90 -lm -lc #ENDIF#
#IBM64_FORTRAN#TAU_FORTRANLIBS = $(TAU_IBM64_FORTRANLIBS) #ENDIF#
#IBM64_FORTRAN#TAU_FORLIBDIR=lib64 #ENDIF#
#IBM_FORTRAN#TAU_FORLIBDIR=lib #ENDIF#
#BGL#TAU_FORLIBDIR=blrts_dev_lib #ENDIF#
#PPC64#TAU_FORTRANLIBS = -L$(EXTRADIR)/$(TAU_FORLIBDIR) -lxlf90 -lxlfmath -lxl #ENDIF#
#BGL#TAU_FORTRANLIBS = -L$(EXTRADIR)/$(TAU_FORLIBDIR) -L$(EXTRADIR)/blrts_lib -lxlf90 -lxlfmath -lxl #ENDIF#

#IBMXLFAPPLE#TAU_FORTRANLIBS	       = -L$(EXTRADIR)/lib -lxlf90 -lxlfmath -lxl      #ENDIF#

#CRAY_FORTRAN#TAU_FORTRANLIBS	       =        #ENDIF#
#CRAY_X1_FORTRAN#TAU_FORTRANLIBS	       =        #ENDIF#
#PGI_FORTRAN#TAU_FORTRANLIBS	       = $(EXTRADIR)/lib/f90main.o -lpgf90 -lpgf90rtl -lpgf90_rpm1 -lpgf902 -lpgftnrtl #ENDIF#
#HP_FORTRAN#TAU_FORTRANLIBS	       = -L$(EXTRADIR)/lib/pa2.0 -lF90 -lcl        #ENDIF#
#INTEL_FORTRAN#TAU_FORTRANLIBS         = -lcprts -lPEPCF90   #ENDIF#
#INTEL32_FORTRAN#TAU_FORTRANLIBS       = -lcprts -lCEPCF90 -lF90 #ENDIF#
#INTELIFORT#TAU_FORTRANLIBS       = -lcprts #ENDIF#
#INTEL81FIX#TAU_FORTRANLIBS       = -L$(TAUGCCLIBDIR) $(TAUGCCLIBOPTS) -lstdc++ $(TAU_GCCLIB) -lcxa -lunwind -L$(EXTRADIR)/lib -lifcore $(EXTRADIR)/lib/for_main.o #ENDIF#
#PGI1.7#LEXTRA = -lstd -lstrm#ENDIF#
#PGI1.7#TAUHELPER = $(TAUROOT)/src/Profile/TauPGIHelper.cpp #ENDIF#
# LINKER OPTIONS
TAU_LINKER_OPT2 = $(LEXTRA)


#ACC#TAUHELPER = -AA #ENDIF#
#FUJITSU_FORTRAN#TAU_FORTRANLIBS	       = $(EXTRADIR)/lib/fj90rt0.o -L$(EXTRADIR)/lib -lfj9i6 -lfj9f6 -lfj9e6        #ENDIF#
#FUJITSU_SOLARIS#TAU_FORTRANLIBS	       = $(EXTRADIR)/lib/fj90rt0.o -L$(EXTRADIR)/lib -lfj90l -lfj90f #ENDIF#
#SUN_FORTRAN#TAU_FORTRANLIBS	       = -lfui -lfsumai -lfprodai -lfminlai -lfmaxlai -lfminvai -lfmaxvai -lfsu -lsunmath        #ENDIF#
#SUN_FORTRAN#TAU_FORTRANLIBS_SUN_OPTERON   = -lfsu -lsunmath #ENDIF#
#SUN386I#TAU_FORTRANLIBS	       = $(TAU_FORTRANLIBS_SUN_OPTERON) #ENDIF#
#SUNX86_64#TAU_FORTRANLIBS	       = $(TAU_FORTRANLIBS_SUN_OPTERON) #ENDIF#
#SOL2#EXTRALIBS = -lsocket -lnsl      #ENDIF#
#SUN386I#EXTRALIBS = -lsocket -lnsl -lrt     #ENDIF#
#SUNX86_64#EXTRALIBS = -lsocket -lnsl -lrt     #ENDIF#
#COMPAQ_FORTRAN#TAU_FORTRANLIBS =  $(EXTRADIR)/lib/cmplrs/fort90/for_main.o -L$(EXTRADIR)/lib -L$(EXTRADIR)/lib/cmplrs/fort90 -L$(EXTRADIR)/lib/cmplrs/fort90 -lUfor -lfor -lFutil -lm -lmld -lexc -lc #ENDIF#
#ABSOFT_FORTRAN#TAU_FORTRANLIBS =  -L$(EXTRADIR)/lib -lfio -lf90math -lU77 -lf77math -lfio #ENDIF#
#LAHEY_FORTRAN#TAU_FORTRANLIBS	       = $(EXTRADIR)/lib/fj90rt0.o -L$(EXTRADIR)/lib -lfj9i6 -lfj9f6 -lfj9e6 -lfccx86_6a #ENDIF#
#HITACHI_FORTRAN#TAU_FORTRANLIBS =  -lf90 -lhf90math #ENDIF#
#NEC_FORTRAN#TAU_FORTRANLIBS =  -f90lib #ENDIF#
#COMPAQ_GUIDEF90#TAU_FORTRANLIBS = $(EXTRADIR)/lib/cmplrs/fort90/for_main.o -lfor #ENDIF#


#HITACHI#TAU_HITACHI_EXTRA 	=  -L/usr/local/lib -llrz32 #ENDIF#

## To use the standard F90 linker instead of TAU_LINKER + TAU_FORTRANLIBS, add
#GNU#TAU_CXXLIBS		= -L$(TAUGCCLIBDIR)  $(TAUGCCLIBOPTS) -lstdc++ $(TAU_GCCLIB) #ENDIF#
#GNU#TAU_GNUCXXLIBS		= -L$(TAUGCCLIBDIR) -lstdc++ $(TAU_GCCLIB) #ENDIF#
#OPEN64ORC#TAU_CXXLIBS		= -lstdc++ #ENDIF#
#PATHSCALE_FORTRAN#TAU_CXXLIBS		= -lstdc++ #ENDIF#
#LAHEY_FORTRAN#TAU_CXXLIBS		= $(TAU_GNUCXXLIBS) /usr/lib/libc.a #ENDIF#
#NAGWARE_FORTRAN#TAU_CXXLIBS		= $(TAU_GNUCXXLIBS) /usr/lib/libc.a #ENDIF#
#PGI#TAU_CXXLIBS        	= -lstd -lC       #ENDIF#
#CRAYX1CC#TAU_CXXLIBS        	= -L/opt/ctl/CC/CC/lib -lC       #ENDIF#

TAU_SGI_INIT = /usr/lib32/c++init.o 
#ENABLE64BIT#TAU_SGI_INIT  = /usr/lib64/c++init.o #ENDIF#
#ENABLEN32BIT#TAU_SGI_INIT = /usr/lib32/c++init.o #ENDIF#
#ENABLE32BIT#TAU_SGI_INIT  = /usr/lib/c++init.o   #ENDIF#

#SGICC#TAU_CXXLIBS      	= $(TAU_SGI_INIT) -lC #ENDIF#
#APPLECXX#TAU_CXXLIBS 		= -lstd -lC #ENDIF#
#SOL2#TAU_CXXLIBS               = -lCstd -lCrun #ENDIF#
#SOL2CC#TAU_CXXLIBS_SUN_OPTERON   = -lCstd -lCrun #ENDIF#
#SUN386I#TAU_CXXLIBS   = $(TAU_CXXLIBS_SUN_OPTERON) #ENDIF#
#SUNX86_64#TAU_CXXLIBS   = $(TAU_CXXLIBS_SUN_OPTERON) #ENDIF#
#FUJITSU_SOLARIS#TAU_CXXLIBS               = -lstd -lstdm #ENDIF#
#PPC64#TAU_XLCLIBS         = -L$(EXTRADIRCXX)/$(TAU_FORLIBDIR) -libmc++ -lstdc++       #ENDIF#
#IBMXLCAPPLE#TAU_FORLIBDIR =lib       #ENDIF#
#IBMXLCAPPLE#TAU_XLCLIBS         = -L$(EXTRADIRCXX)/$(TAU_FORLIBDIR) -libmc++ -lstdc++       #ENDIF#
#BGL#TAU_XLCLIBS         = -L$(EXTRADIRCXX)/blrts_dev_lib -L$(EXTRADIRCXX)/blrts_lib -libmc++ -L/bgl/BlueLight/ppcfloor/blrts-gnu/powerpc-bgl-blrts-gnu/lib -lstdc++      #ENDIF#
#SP1#TAU_XLCLIBS         = -lC            #ENDIF#
#USE_IBMXLC#TAU_CXXLIBS         = $(TAU_XLCLIBS) #ENDIF#
#USE_DECCXX#TAU_CXXLIBS         = -lcxxstd -lcxx #ENDIF#
#USE_INTELCXX#TAU_CXXLIBS_INTEL        = -lcprts -lPEPCF90 #ENDIF#
#USE_INTELCXX#TAU_CXXLIBS_INTEL        = -lcprts #ENDIF#
#INTELIFORT#TAU_CXXLIBS_INTEL        = -lcprts #ENDIF#
#INTEL81FIX#TAU_CXXLIBS_INTEL        = -L$(TAUGCCLIBDIR) $(TAUGCCLIBOPTS) -lstdc++ $(TAU_GCCLIB) -lcxa -lunwind#ENDIF#
#USE_INTELCXX#TAU_CXXLIBS        = $(TAU_CXXLIBS_INTEL) #ENDIF#
#APPLECXX#TAU_CXXLIBS 		= -lstdc++ -L$(TAUGCCLIBDIR)  $(TAUGCCLIBOPTS) -lgcc_s -lSystemStubs #ENDIF#

# EXTERNAL PACKAGES: EPILOG
#EPILOG#TAU_LINKER_OPT3 	=  -L$(EPILOGDIR)/lib -L$(EPILOGDIR)/src -lelg $(TAU_HITACHI_EXTRA) #ENDIF#
#EPILOGMPI#TAU_LINKER_OPT3 	=  -L$(EPILOGDIR)/lib -L$(EPILOGDIR)/src -lelg.mpi  $(TAU_HITACHI_EXTRA) #ENDIF#
#EPILOGOMPI#TAU_LINKER_OPT3 	=  -L$(EPILOGDIR)/lib -L$(EPILOGDIR)/src -lelg.ompi $(TAU_HITACHI_EXTRA) #ENDIF#
#EPILOGOMP#TAU_LINKER_OPT3 	=  -L$(EPILOGDIR)/lib -L$(EPILOGDIR)/src -lelg.omp  $(TAU_HITACHI_EXTRA) #ENDIF#

#HITACHI_OPENMP#TAU_LINKER_OPT4 = -lcompas -lpthreads -lc_r #ENDIF#
#OPENMP#TAU_LINKER_OPT5 = -mp #ENDIF#
#SOL2CC_OPENMP#TAU_LINKER_OPT5 = #ENDIF#
#GNU#TAU_LINKER_OPT5 = #ENDIF#
#COMPAQCXX_OPENMP#TAU_LINKER_OPT5 = -omp #ENDIF#
#IBMXLC_OPENMP#TAU_LINKER_OPT5 = -qsmp=omp #ENDIF#
#OPEN64_OPENMP#TAU_LINKER_OPT5 = -mp #ENDIF#
#GUIDE#TAU_LINKER_OPT5 = #ENDIF#
#PGIOPENMP#TAU_LINKER_OPT5 = -mp #ENDIF#
#INTELOPENMP#TAU_LINKER_OPT5 = -openmp #ENDIF#

# MALLINFO needs -lmalloc on sgi, sun 
#SGIMP#TAU_LINKER_OPT6 = -lmalloc #ENDIF#
#SOL2#TAU_LINKER_OPT6 = #ENDIF#
#SUN386I#TAU_LINKER_OPT6 = -lmalloc #ENDIF#
#SUNX86_64#TAU_LINKER_OPT6 = -lmalloc #ENDIF#

# We need -lCio with SGI CC 7.4+
#SGICC#TAU_LINKER_OPT7 = -lCio #ENDIF#

# charm
#TAU_CHARM#TAU_LINKER_OPT8 = -lconv-core #ENDIF#

# extra libs
#SUN386I#TAU_LINKER_OPT9 = $(EXTRALIBS) #ENDIF#
#SUNX86_64#TAU_LINKER_OPT9 = $(EXTRALIBS) #ENDIF#

#USE_IBMXLC#TAU_IBM_PYTHON_SHFLAG = -Wl,-bI:$(PYTHON_LIBDIR)/config/python.exp -Wl,-einitpytau#ENDIF#
#GNU#TAU_IBM_PYTHON_SHFLAG = -Wl,-bI:$(PYTHON_LIBDIR)/config/python.exp -Wl,-einitpytau#ENDIF#
#KAI#TAU_IBM_PYTHON_SHFLAG = --backend -Wl,-bI:$(PYTHON_LIBDIR)/config/python.exp --backend -Wl,-einitpytau#ENDIF#
#ACC#TAU_HPUX_PYTHON_SHFLAG = -lstd_v2 -lCsup_v2 -lm -lcl -lc #ENDIF#

#USE_IBMXLC#TAU_IBM_LD_FLAGS = -binitfini:poe_remote_main -brtl #ENDIF#
#GNU#TAU_IBM_LD_FLAGS = -Wl,-binitfini:poe_remote_main -Wl,-brtl #ENDIF#
#KAI#TAU_IBM_LD_FLAGS = --backend -binitfini:poe_remote_main --backend -brtl #ENDIF#


#PYTHON#TAU_IBM_SHFLAGS = $(TAU_IBM_PYTHON_SHFLAG) #ENDIF#
#PYTHON#TAU_HPUX_SHFLAGS = $(TAU_HPUX_PYTHON_SHFLAG) #ENDIF#
#SP1#TAU_EXTRA_LIBRARY_FLAGS = $(TAU_IBM_SHFLAGS)  #ENDIF#
#SOL2#TAU_EXTRA_LIBRARY_FLAGS = #ENDIF#
#SGIMP#TAU_EXTRA_LIBRARY_FLAGS = -lmalloc  #ENDIF#
#HP#TAU_EXTRA_LIBRARY_FLAGS  = $(TAU_HPUX_SHFLAGS) #ENDIF#

#MPI#TAU_MPI_WRAPPER_LIB= -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -lTauMpi$(TAU_CONFIG) #ENDIF#
#EPILOGMPI#TAU_MPI_WRAPPER_LIB= #ENDIF#
#EPILOGOMPI#TAU_MPI_WRAPPER_LIB= #ENDIF#

##############################################
# Build TAU_LINKER_SHOPTS
#GNU#TAU_IBM_LINKER_SHOPTS=-Wl,-brtl -Wl,-bexpall #ENDIF#
#USE_IBMXLC#TAU_IBM_LINKER_SHOPTS= -brtl -bexpall #ENDIF#
#KAI#TAU_IBM_LINKER_SHOPTS= --backend -brtl #ENDIF#
#SP1#TAU_LINKER_SHOPTS= $(TAU_IBM_LINKER_SHOPTS) #ENDIF#

##############################################
# MPI _r suffix check (as in libmpi_r)
#MPI_R_SUFFIX#TAU_MPI_R_SUFFIX=_r    #ENDIF#

##############################################
# Flags to build a shared object: TAU_SHFLAGS
#GNU#AR_SHFLAGS                 = -shared       #ENDIF#
#PGI#AR_SHFLAGS                 = -shared       #ENDIF#
#SGICC#AR_SHFLAGS               = -shared       #ENDIF#
#APPLECXX#AR_SHFLAGS = -dynamiclib -flat_namespace -undefined suppress #ENDIF#
#SOL2#AR_SHFLAGS                = -G            #ENDIF#
#SUN386I#AR_SHFLAGS                = -G            #ENDIF#
#SUNX86_64#AR_SHFLAGS                = -G            #ENDIF#
#USE_IBMXLC#AR_SHFLAGS          = -G                #ENDIF#
#USE_DECCXX#AR_SHFLAGS          = -shared               #ENDIF#
#USE_INTELCXX#AR_SHFLAGS        = -shared               #ENDIF#
#ACC#AR_SHFLAGS                 = -b            #ENDIF#
TAU_SHFLAGS = $(AR_SHFLAGS) -o

############# RANLIB Options #############
TAU_RANLIB = echo "Built" 
#APPLECXX#TAU_RANLIB      = ranlib  #ENDIF#
#IBMXLCAPPLE#TAU_RANLIB      = ranlib  #ENDIF#

##############################################
TAU_AR                 = ar             #ENDIF#
#IBM64#TAU_AR          = ar -X64        #ENDIF#
#PPC64#TAU_AR          = ar 		#ENDIF#
#IBM64LINUX#TAU_AR     = ar             #ENDIF#


##############################################
# PDT OPTIONS
PDTARCHDIR=$(TAU_ARCH)
#PDTARCH#PDTARCHDIR=$(TAU_ARCH)/$(PDTCOMPDIR)#ENDIF#


##############################################

PROFILEOPTS = $(PROFILEOPT1)  $(PROFILEOPT2)  $(PROFILEOPT3)  $(PROFILEOPT4)  \
              $(PROFILEOPT5)  $(PROFILEOPT6)  $(PROFILEOPT7)  $(PROFILEOPT8)  \
	      $(PROFILEOPT9)  $(PROFILEOPT10) $(PROFILEOPT11) $(PROFILEOPT12) \
	      $(PROFILEOPT13) $(PROFILEOPT14) $(PROFILEOPT15) $(PROFILEOPT16) \
	      $(PROFILEOPT17) $(PROFILEOPT18) $(PROFILEOPT19) $(PROFILEOPT20) \
	      $(PROFILEOPT21) $(PROFILEOPT22) $(PROFILEOPT23) $(PROFILEOPT24) \
	      $(PROFILEOPT25) $(PROFILEOPT26) $(PROFILEOPT27) $(PROFILEOPT28) \
	      $(PROFILEOPT29) $(PROFILEOPT30) $(PROFILEOPT31) $(PROFILEOPT32) \
	      $(PROFILEOPT33) $(PROFILEOPT34) $(PROFILEOPT35) $(PROFILEOPT36) \
              $(PROFILEOPT37) $(PROFILEOPT38) $(PROFILEOPT39) $(PROFILEOPT40) \
              $(PROFILEOPT41) $(PROFILEOPT42) $(PROFILEOPT43) $(PROFILEOPT44) \
              $(PROFILEOPT45) $(PROFILEOPT46) $(PROFILEOPT47) $(PROFILEOPT48) \
              $(PROFILEOPT49) $(PROFILEOPT50) $(PROFILEOPT51) $(PROFILEOPT52) \
              $(PROFILEOPT53) $(PROFILEOPT54) $(PROFILEOPT55) $(PROFILEOPT56) \
	      $(PROFILEOPT57) $(PROFILEOPT58) $(PROFILEOPT59) $(PROFILEOPT60) \
	      $(PROFILEOPT61) $(PROFILEOPT62) $(PROFILEOPT63) $(PROFILEOPT64) \
	      $(PROFILEOPT65) $(PROFILEOPT66) $(PROFILEOPT67) $(TRACEOPT)

##############################################

TAU_LINKER_OPTS = $(TAU_LINKER_OPT1) $(TAU_LINKER_OPT2) $(TAU_LINKER_OPT3) \
                  $(TAU_LINKER_OPT4) $(TAU_LINKER_OPT5) $(TAU_LINKER_OPT6) \
                  $(TAU_LINKER_OPT7) $(TAU_LINKER_OPT8) $(TAU_LINKER_OPT9)

##############################################

############# TAU Fortran ####################
TAU_LINKER=$(TAU_CXX)
#INTEL_FORTRAN##TAU_LINKER=$(TAU_F90) #ENDIF#
#INTEL32_FORTRAN##TAU_LINKER=$(TAU_F90) #ENDIF#
# Intel efc compiler acts as a linker - NO. Let C++ be the linker. 

##############################################
############# TAU Options ####################
TAUDEFS 	= $(PROFILEOPTS) 

TAUINC        	= -I$(TAU_PREFIX_INSTALL_DIR)/include

TAULIBS		= $(TAUHELPER) -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -ltau$(TAU_CONFIG)  $(TAU_LINKER_OPTS)

TAUMPILIBS	= $(TAU_MPI_LIB)

TAUMPIFLIBS	= $(TAU_MPI_FLIB)

### ACL S/W requirement
TAU_DEFS	= $(TAUDEFS)

TAU_INCLUDE 	= -I$(TAU_PREFIX_INSTALL_DIR)/include

TAU_INCLUDE_MEMORY = -I$(TAU_PREFIX_INSTALL_DIR)/include/Memory
#IBMXLCAPPLE#TAU_INCLUDE_MEMORY = -I$(TAU_PREFIX_INSTALL_DIR)/include/MemoryWrapper #ENDIF#
#APPLECXX#TAU_INCLUDE_MEMORY = -I$(TAU_PREFIX_INSTALL_DIR)/include/MemoryWrapper #ENDIF#

TAU_LIBS	= $(TAUHELPER) -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -ltau$(TAU_CONFIG)   $(TAU_LINKER_OPTS) 

TAU_SHLIBS	= $(TAUHELPER) -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -lTAUsh$(TAU_CONFIG) $(TAU_LINKER_OPTS) $(TAU_LINKER_SHOPTS) 

TAU_SHLIBS_NOSHOPTS = $(TAUHELPER) -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -lTAUsh$(TAU_CONFIG) $(TAU_LINKER_OPTS)

TAU_DISABLE	= $(TAUHELPER) -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -lTauDisable

TAU_MPI_INCLUDE	= $(TAU_MPI_INC)

TAU_MPI_LIBS	= $(TAU_MPI_LIB)

TAU_MPI_FLIBS	= $(TAU_MPI_FLIB)

## TAU TRACE INPUT LIBRARY (can build a trace converter using TAU TIL)
TAU_TRACE_INPUT_LIB = -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -lTAU_traceinput$(TAU_CONFIG)

## Don't include -lpthread or -lsmarts. Let app. do that. 
#############################################
## IBM SPECIFIC CHANGES TO TAU_MPI_LIBS
#SP1#TAU_MPI_LDFLAGS  = $(TAU_IBM_LD_FLAGS) #ENDIF#
#MPI#TAU_LDFLAGS  = $(TAU_MPI_LDFLAGS) #ENDIF#
#SP1#TAU_IBM_MPI_LIBS    = $(TAU_MPI_LIB)  -L$(TAU_MPILIB_DIR)/ip  -lvtd$(TAU_MPI_R_SUFFIX) #ENDIF#
#SP1#TAU_IBM_FMPI_LIBS    = $(TAU_MPI_FLIB)  -L$(TAU_MPILIB_DIR)/ip  -lvtd$(TAU_MPI_R_SUFFIX) #ENDIF#
#SP1#TAU_MPI_LIBS_FLAGS= $(TAU_IBM_MPI_LIBS) #ENDIF#
#SP1#TAU_MPI_FLIBS_FLAGS   = $(TAU_IBM_MPI_FLIBS) #ENDIF#
#MPI#TAU_MPI_LIBS_FLAGS = $(TAU_MPI_LIB) #ENDIF#
#MPI#TAU_MPI_FLIBS_FLAGS  = $(TAU_MPI_FLIB) #ENDIF#
#MPI#TAU_MPI_LIBS = $(TAU_MPI_LIBS_FLAGS) #ENDIF#
#MPI#TAU_MPI_FLIBS  = $(TAU_MPI_FLIBS_FLAGS) #ENDIF#

#SP1#TAUMPILIBS      = $(TAU_MPI_LIBS) 	#ENDIF#
#SP1#TAUMPIFLIBS     = $(TAU_MPI_FLIBS)	#ENDIF#
#############################################
#SHMEM#TAU_SHMEM_OBJS		     = TauShmemCray.o	  #ENDIF#
#SP1#TAU_SHMEM_OBJS		     = TauShmemTurbo.o	  #ENDIF#
#GPSHMEM#TAU_SHMEM_OBJS		     = TauShmemGpshmem.o  #ENDIF#

TAU_SHMEM_INCLUDE	= $(TAU_SHMEM_INC)

TAU_SHMEM_LIBS	= -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/ -lTauShmem$(TAU_CONFIG) $(TAU_SHMEM_LIB)
#############################################
# TAU COMPILER SHELL SCRIPT OPTIONS
TAUCOMPILEROPTS=  -optPdtDir="$(PDTDIR)/${PDTARCHDIR}"\
        -optPdtCOpts="$(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE)"\
        -optPdtCxxOpts="$(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE)"\
        -optTauInstr="$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_instrumentor" \
        -optNoMpi \
	-optOpariDir="$(OPARIDIR)" -optOpariTool="$(TAU_OPARI_TOOL)" \
	-optTauCC="$(TAU_CC)" \
	-optTauIncludes="$(TAU_INCLUDE) $(TAU_MPI_INCLUDE)" \
	-optTauDefs="$(TAU_DEFS)" \
        -optTauCompile="$(TAU_INCLUDE) $(TAU_MPI_INCLUDE) $(TAU_DEFS) "\
        -optLinking="$(TAU_MPI_FLIBS) $(TAU_LIBS) $(TAU_LDFLAGS) $(TAU_CXXLIBS)"
#############################################
TAU_COMPILER=$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_compiler.sh $(TAUCOMPILEROPTS)
#############################################
# These options could be included in the application Makefile as 
#CFLAGS        	= $(TAUDEFS) $(TAUINC)
#
#LIBS 	      	= $(TAULIBS)
#
# To run the application without Profiling/Tracing use
#CFLAGS		= $(TAUINC)   
# Don't use TAUDEFS but do include TAUINC
# Also ignore TAULIBS when Profiling/Tracing is not used.
#############################################

