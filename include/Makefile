#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.acl.lanl.gov/tau		           **
#****************************************************************************
#*    Copyright 1997, 1998, 1999 				   	   **
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
EXTRADIR=
TAUROOT=
TULIPDIR=
PCLDIR=
PAPIDIR=
PDTDIR=
DYNINSTDIR=
JDKDIR=
OPARIDIR=
TAU_OPTIONS=
TAU_MPI_INC=
TAU_MPI_LIB=
TAU_MPI_FLIB=
TAU_PREFIX_INSTALL_DIR=
#######################################################################
 
#ENABLE64BIT#ABI 	     = -64		  #ENDIF#
#ENABLEN32BIT#ABI 	     = -n32		  #ENDIF#
#ENABLE32BIT#ABI 	     = -32		  #ENDIF#

############# Standard Defines ##############
TAU_CC = $(CONFIG_CC) $(ABI) $(ISA)
TAU_CXX = $(CONFIG_CXX) $(ABI) $(ISA) $(USER_OPT)
DEFINES =
INSTALL = /bin/cp
SHELL = /bin/sh
LSX = .a
#############################################
# JAVA DEFAULT ARCH 
#############################################
JDKARCH                 = linux
#DEC_ALPHA#JDKARCH      = alpha      #ENDIF#
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

#############################################
# Opari OBJECTS
#############################################
#OPARI#OPARI_O 	= TauOpari.o #ENDIF#

#############################################
# DYNINST DEFAULT ARCH
#############################################
DYNINST_PLATFORM 	= $(PLATFORM) 


#PCL#include $(TAUROOT)/include/makefiles/PCLMakefile.stub	  #ENDIF#

############# Fortran Compiler #############
#GNU_FORTRAN#TAU_F90         = g77      $(ABI)    #ENDIF#
#SGI_FORTRAN#TAU_F90         = f90      $(ABI)    #ENDIF#
#IBM_FORTRAN#TAU_F90         = xlf90    $(ABI)    #ENDIF#
#IBM64_FORTRAN#TAU_F90       = xlf90    $(ABI)    #ENDIF#
#CRAY_FORTRAN#TAU_F90        = f90      $(ABI)    #ENDIF#
#PGI_FORTRAN#TAU_F90         = pgf90    $(ABI)    #ENDIF#
#ABSOFT_FORTRAN#TAU_F90      = f90      $(ABI)    #ENDIF#
#FUJITSU_FORTRAN#TAU_F90     = F90      $(ABI)    #ENDIF#
#SUN_FORTRAN#TAU_F90         = f90      $(ABI)    #ENDIF#
#COMPAQ_FORTRAN#TAU_F90      = f90      $(ABI)    #ENDIF#
#KAI_FORTRAN#TAU_F90         = guidef90 $(ABI)    #ENDIF#


############# Profiling Options #############
#PROFILE#PROFILEOPT1           = -DPROFILING_ON        #ENDIF#
#SGITIMERS#PROFILEOPT2         = -DSGI_TIMERS          #ENDIF#
#CPUTIME#PROFILEOPT2           = -DCPU_TIME          #ENDIF#
#PAPIWALLCLOCK#PROFILEOPT2     = -DTAU_PAPI_WALLCLOCKTIME    #ENDIF#
#PAPIVIRTUAL#PROFILEOPT2       = -DTAU_PAPI_VIRTUAL    #ENDIF#
#SGICOUNTERS#PROFILEOPT3       = -DSGI_HW_COUNTERS     #ENDIF#
#PCL#PROFILEOPT3       	       = -DTAU_PCL -D$(TARGET_SYSTEM) -I$(PCLDIR)/src #ENDIF#
#PAPI#PROFILEOPT3              = -DTAU_PAPI -I$(PAPIDIR)/src #ENDIF#
#PCL#PCL_O       	       = PclLayer.o 	       #ENDIF#
#PAPI#PAPI_O       	       = PapiLayer.o 	       #ENDIF#
#PROFILECALLS#PROFILEOPT4      = -DPROFILE_CALLS       #ENDIF#
#PROFILESTATS#PROFILEOPT5      = -DPROFILE_STATS       #ENDIF#
#DEBUGPROF#PROFILEOPT6         = -DDEBUG_PROF          #ENDIF#
#STDCXXLIB#PROFILEOPT7         = -DTAU_STDCXXLIB       #ENDIF#
#INTELTFLOP#PROFILEOPT8        = -DPOOMA_TFLOP         #ENDIF#
#NORTTI#PROFILEOPT9            = -DNO_RTTI             #ENDIF#
#RTTI#PROFILEOPT9              = -DRTTI             #ENDIF#
#GNU#PROFILEOPT9           = -DNO_RTTI             #ENDIF#
#GNU#PROFILEOPT10              = -DTAU_GNU           #ENDIF#
#CALLSTACK#PROFILEOPT11        = -DPROFILE_CALLSTACK   #ENDIF#
#PGI1.7#PROFILEOPT12	       = -DPGI 		       #ENDIF#
#CRAYKAI#PROFILEOPT12	       = -DCRAYKAI	       #ENDIF#
#CRAYCC#PROFILEOPT13	       = -h instantiate=used -DCRAYCC   #ENDIF#
#SGICC##PROFILEOPT14	       = -ptused 	       #ENDIF#
#PTHREAD_AVAILABLE#PROFILEOPT15 = -DPTHREADS  	       #ENDIF#
#TAU_SPROC#PROFILEOPT15 	= -DTAU_SPROC  	       #ENDIF#
#TULIPTHREADS#PROFILEOPT16 = -DTULIPTHREADS            #ENDIF#
#TRACE#TRACEOPT		       = -DTRACING_ON	       #ENDIF#
#TRACE#EVENTS_O                = Tracer.o              #ENDIF#
#PTHREAD_AVAILABLE#THR_O       = PthreadLayer.o        #ENDIF#
#TAU_SPROC#THR_O       	       = SprocLayer.o        #ENDIF#
#JAVA#THR_O       	       = JavaThreadLayer.o     #ENDIF#
#TULIPTHREADS#THR_O       = TulipThreadLayer.o         #ENDIF#
#TULIPTHREADS#PROFILEOPT17  = -I$(TULIPDIR)/include -I$(TULIPDIR)/Tuliplib   #ENDIF#
#SMARTS#PROFILEOPT17  = -I$(TULIPDIR)/include -I$(TULIPDIR)/lib -I$(TULIPDIR)/machine-specific/$(HOSTTYPE)  #ENDIF#
#SMARTS#PROFILEOPT18  = -DSMARTS   #ENDIF#
#KAI#PROFILEOPT19  	       = -DKAI   #ENDIF#
#USE_DECCXX#PROFILEOPT19  	       = -DTAU_DOT_H_LESS_HEADERS   #ENDIF#
#PGI#PROFILEOPT19  	       = -DTAU_DOT_H_LESS_HEADERS   #ENDIF#
#FUJITSU#PROFILEOPT19  	       = -DFUJITSU -DTAU_DOT_H_LESS_HEADERS   #ENDIF#
#KAINOEX#PROFILEOPT20  	       = --no_exceptions   #ENDIF#
#SGICCNOEX#PROFILEOPT20  	       = -LANG:exceptions=off  #ENDIF#
#SP1#PROFILEOPT21	       = -D_POSIX_SOURCE -D_XOPEN_SOURCE -DTAU_AIX #ENDIF#
#USE_IBMXLC#PROFILEOPT21	       = -DTAU_DOT_H_LESS_HEADERS -DTAU_XLC#ENDIF#
#PCLPTHREAD#PROFILEOPT22	       = -DPCL_MUTEX_LOCK #ENDIF#
#JAVA#PROFILEOPT23	       = -DJAVA 		#ENDIF#
#MONITOR#PROFILEOPT24	       = -DMONITORING_ON 	#ENDIF#
#JAVA#PROFILEOPT25 = -I$(JDKDIR)/include -I$(JDKDIR)/include/$(JDKARCH) #ENDIF#
#MPI#PROFILEOPT26 = -DTAU_MPI #ENDIF#
#MPI_THREADED#PROFILEOPT26 = -DTAU_MPI -DTAU_MPI_THREADED #ENDIF#
#OPENMP#PROFILEOPT27 = -mp -D_OPENMP -DTAU_OPENMP#ENDIF#
#COMPAQCXX_OPENMP#PROFILEOPT27 = -omp -DTAU_OPENMP#ENDIF#
#IBMXLC_OPENMP#PROFILEOPT27 = -qsmp=omp -D_OPENMP -DTAU_OPENMP #ENDIF#
#GUIDE#PROFILEOPT27 = -DTAU_OPENMP #ENDIF#
#PGIOPENMP#PROFILEOPT27 = -mp -D_OPENMP -DTAU_OPENMP -U_RWSTD_MULTI_THREAD -U_REENTRANT #ENDIF#
#OPARI#PROFILEOPT28 = -I$(OPARIDIR)/lib #ENDIF#

#FOR F90 MPI on SGI
#SGIMP#FWRAPPER	       = TauFMpi.o       #ENDIF#
#SP1#FWRAPPER	       = TauFMpi.o       #ENDIF#

#SGICOUNTERS#LEXTRA	       = -lperfex 	       #ENDIF#
#PCL#LEXTRA	       	       = $(LOADLIBS) 	       #ENDIF#
#PAPI#LEXTRA	       	       = $(PAPIDIR)/src/libpapi.a #ENDIF#
#SOL2PAPI#LEXTRA	       = $(PAPIDIR)/src/libpapi.a -lcpc #ENDIF#
#IBMPAPI#LEXTRA  = $(PAPIDIR)/src/libpapi.a -L/usr/lpp/pmtoolkit/lib -lpmapi#ENDIF#
#PTHREAD_AVAILABLE#LEXTRA1     = -lpthread	       #ENDIF#
#TULIPTHREADS#LEXTRA1     = -L$(TULIPDIR)/Tuliplib  -ltulip	       #ENDIF#
#SMARTS##include $(TAUROOT)/include/makefiles/GNUmakefile-$(HOSTTYPE)  #ENDIF#
#SMARTS#LEXTRA1	    = $(LSMARTS)	#ENDIF#
#GNU_FORTRAN#TAU_FORTRANLIBS	       = -lg2c       #ENDIF#
#SGI_FORTRAN#TAU_FORTRANLIBS	       = -lfortran -lftn       #ENDIF#
#IBM_FORTRAN#TAU_FORTRANLIBS	       =  -bh:4 -bpT:0x10000000 -bpD:0x20000000 /lib/crt0.o -lxlf90 -lm -lc #ENDIF#
#IBM64_FORTRAN#TAU_FORTRANLIBS      =  -bh:4 -bpT:0x10000000 -bpD:0x20000000 -b64 /lib/crt0_64.o -lxlf90 -lm -lc #ENDIF#

#CRAY_FORTRAN#TAU_FORTRANLIBS	       =        #ENDIF#
#PGI_FORTRAN#TAU_FORTRANLIBS	       = $(EXTRADIR)/lib/f90main.o -lpgf90 -lpgf90rtl -lpgf90_rpm1 -lpgf902        #ENDIF#
#PGI1.7#LEXTRA = -lstd -lstrm#ENDIF#
#PGI1.7#TAUHELPER = $(TAUROOT)/src/Profile/TauPGIHelper.cpp #ENDIF#
#FUJITSU_FORTRAN#TAU_FORTRANLIBS	       = $(EXTRADIR)/lib/fj90rt0.o -L$(EXTRADIR)/lib -lfj9i6 -lfj9f6 -lfj9e6        #ENDIF#
#SUN_FORTRAN#TAU_FORTRANLIBS	       = -lfui -lfai -lfai2 -lfsumai -lfprodai -lfminlai -lfmaxlai -lfminvai -lfmaxvai -lfsu -lsunmath        #ENDIF#
#SOL2#EXTRALIBS = -lsocket -lnsl      #ENDIF#
#COMPAQ_FORTRAN#TAU_FORTRANLIBS =  $(EXTRADIR)/lib/cmplrs/fort90/for_main.o -L$(EXTRADIR)/lib -L$(EXTRADIR)/lib/cmplrs/fort90 -L$(EXTRADIR)/lib/cmplrs/fort90 -lUfor -lfor -lFutil -lm -lmld -lexc -lc #ENDIF#

#MPI#TAU_MPI_WRAPPER_LIB= -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -lTauMpi$(TAU_OPTIONS) #ENDIF#

PROFILEOPTS = $(PROFILEOPT1)  $(PROFILEOPT2)  $(PROFILEOPT3)  $(PROFILEOPT4)  \
              $(PROFILEOPT5)  $(PROFILEOPT6)  $(PROFILEOPT7)  $(PROFILEOPT8)  \
	      $(PROFILEOPT9)  $(PROFILEOPT10) $(PROFILEOPT11) $(PROFILEOPT12) \
	      $(PROFILEOPT13) $(PROFILEOPT14) $(PROFILEOPT15) $(PROFILEOPT16) \
	      $(PROFILEOPT17) $(PROFILEOPT18) $(PROFILEOPT19) $(PROFILEOPT20) \
	      $(PROFILEOPT21) $(PROFILEOPT22) $(PROFILEOPT23) $(PROFILEOPT24) \
	      $(PROFILEOPT25) $(PROFILEOPT26) $(PROFILEOPT27) $(PROFILEOPT28) \
              $(TRACEOPT)
##############################################

############# TAU Options ####################
TAUDEFS 	= $(PROFILEOPTS) 

TAUINC        	= -I$(TAUROOT)/include

TAULIBS		= $(TAUHELPER) -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -ltau$(TAU_OPTIONS) $(LEXTRA) 

TAUMPILIBS	= $(TAU_MPI_LIB)

TAUMPIFLIBS	= $(TAU_MPI_FLIB)

### ACL S/W requirement
TAU_DEFS	= $(TAUDEFS)

TAU_INCLUDE 	= -I$(TAUROOT)/include

TAU_LIBS	= $(TAUHELPER) -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -ltau$(TAU_OPTIONS) $(LEXTRA)

TAU_SHLIBS	= $(TAUHELPER) -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -lTAUsh$(TAU_OPTIONS) $(LEXTRA)

#USE_IBMXLC#TAU_SHLIBS	= $(TAUHELPER) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/libTAUsh$(TAU_OPTIONS).so $(LEXTRA) #ENDIF#

TAU_DISABLE	= $(TAUHELPER) -L$(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib -lTauDisable

TAU_MPI_INCLUDE	= $(TAU_MPI_INC)

TAU_MPI_LIBS	= $(TAU_MPI_LIB)

TAU_MPI_FLIBS	= $(TAU_MPI_FLIB)
## Don't include -lpthread or -lsmarts. Let app. do that. 
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
