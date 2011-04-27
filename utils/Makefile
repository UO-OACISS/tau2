#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/paracomp/tau    **
#****************************************************************************
#*    Copyright 1997  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
#######################################################################
##                  pC++/Sage++  Copyright (C) 1993,1995             ##
##  Indiana University  University of Oregon  University of Rennes   ##
#######################################################################
 
include ../include/Makefile
########### Automatically modified by the configure script ############
CONFIG_ARCH=default
CONFIG_CC=gcc
CONFIG_CXX=g++
PDT_CXX=
USER_OPT=-g
TAU_PREFIX_INSTALL_DIR=
VTFDIR=
OTFDIR=
PDTDIR=
DYNINSTDIR=
#######################################################################
 
#MIPSR8K#ISA          = -mips4		   #ENDIF#

############# Standard Defines ##############
CC = $(TAU_CC_FE) $(ABI) $(ISA)
CXX = $(TAU_CXX_FE) $(ABI) $(ISA)
#CC = $(CONFIG_CC) $(ABI) $(ISA)
#CXX = $(CONFIG_CXX) $(ABI) $(ISA)
#CRAYXMT#CC = $(TAU_CC_FE) $(ABI) $(ISA) #ENDIF#
#CRAYXMT#CXX = $(TAU_CXX_FE) $(ABI) $(ISA) #ENDIF#
TAU_INSTALL = /bin/cp
TAU_SHELL = /bin/sh
LSX = .a
#############################################

LINKER	= $(CC)

#INTELCXXLIBICC#INTELOPTS=-cxxlib-icc #ENDIF#
#INTELCXXLIBICC#CXX=$(CONFIG_CXX) $(ABI) $(ISA) $(INTELOPTS) #ENDIF#

LEXTRA = 
DWARFOPTS=
#APPLECXX#CEXTRA = -DAPPLECXX #ENDIF#
#HP_CFLAGS#CEXTRA = -AA -DTAU_DOT_H_LESS_HEADERS #ENDIF#
#CRAYCC#CEXTRA = -DTAU_DOT_H_LESS_HEADERS #ENDIF#
#CRAYX1CC#CEXTRA = -DTAU_DOT_H_LESS_HEADERS #ENDIF#
#HPGNU#CEXTRA = -DTAU_HP_GNU #ENDIF#
#HP_CFLAGS#LSX = .sl#ENDIF#
#SYMMETRY#LEXTRA = -lseq#ENDIF#
#PTX#LEXTRA = -lseq#ENDIF#
#SGICC#LEXTRA = -lCio #ENDIF#
#SP1#CEXTRA = -D__SP1__ -D_ANY_SOURCE #ENDIF#
#USE_IBMXLC#CEXTRA = -DTAU_DOT_H_LESS_HEADERS -DTAU_XLC #ENDIF#
#USE_DECCXX#CEXTRA = -DTAU_DOT_H_LESS_HEADERS -x c++ -D__USE_STD_IOSTREAM #ENDIF#
#USE_INTELCXX#CEXTRA = -DTAU_DOT_H_LESS_HEADERS -DTAU_INTEL #ENDIF#
#USE_NECCXX#CEXTRA = -DTAU_DOT_H_LESS_HEADERS -DTAU_NEC -Xp #ENDIF#
#CM5#CEXTRA = -D__CM5__#ENDIF#
#SGICC#CEXTRA = -LANG:std -D__SGIMP__#ENDIF#
#SGICC#CEXTRA = -D_STANDARD_C_PLUS_PLUS -D__SGIMP__#ENDIF#
WARNINGSOPT	= -w 
#CRAYX1CC#WARNINGSOPT	= #ENDIF#
#CRAYCC#WARNINGSOPT= -DCRAYCC  #ENDIF#
#PGI1.7#LEXTRA = -lstd -lstrm  #ENDIF#
#PGI#CEXTRA = -DPGI -DTAU_DOT_H_LESS_HEADERS  #ENDIF#
#CRAYCNL#CEXTRA = -DPGI -DTAU_DOT_H_LESS_HEADERS  #ENDIF#
#CATAMOUNT#CEXTRA = -DTAU_DOT_H_LESS_HEADERS  #ENDIF#
#FUJITSU#CEXTRA = -DFUJITSU -DTAU_DOT_H_LESS_HEADERS  #ENDIF#
CWARNING	= $(WARNINGSOPT)
#PDT#PDTOPTS 	= -L$(PDTDIR)/$(PDTARCHDIR)/lib -lpdb #ENDIF#
#PDT#INSTR 		= tau_instrumentor #ENDIF#
#PDT#WRAP 		= tau_wrap #ENDIF#
#PDT#PDTOMPCHECK	= tau_ompcheck #ENDIF#
#PDT#HEADERLIST 	= tau_header_list #ENDIF#
SELECTFILE 	= tau_selectfile
#OPARI#TAUOMPCHECK    = $(PDTOMPCHECK) #ENDIF#
#DYNINST#TAURUN 	= tau_run #ENDIF#
#TRACE#TRACE2PROFILE	= trace2profile/trace2profile #ENDIF#
#TRACE#INSTALLTRACE2PROFILE	= $(INSTALLDEST)/trace2profile #ENDIF#
#TRACE#TAU_TIMECORRECT	= tau_timecorrect #ENDIF#
#TRACE#INSTALLTAU_TIMECORRECT	= $(INSTALLDEST)/tau_timecorrect #ENDIF#
#VTF#TAU2VTF 	= tau2vtf #ENDIF#
#OTF#TAU2OTF 	= tau2otf #ENDIF#
#VTF#VTF2PROFILE 	= vtfconverter/vtf2profile #ENDIF#
#VTF#INSTALLTAU2VTF 	= $(INSTALLDEST)/tau2vtf #ENDIF#
#VTF#INSTALLVTF2PROFILE 	= $(INSTALLDEST)/vtf2profile #ENDIF#
#OTF#INSTALLTAU2OTF 	= $(INSTALLDEST)/tau2otf #ENDIF#
#OTF#OTFLIBS 	= -L$(OTFDIR)/$(TAU_ARCH)/lib -lotf $(TAU_TRACE_INPUT_LIB) #ENDIF#
#OTFSRC#OTFLIBS 	= -L$(OTFDIR)/lib -lotf -lz $(TAU_TRACE_INPUT_LIB) #ENDIF#
#EBS2OTF#INSTALL_EBS2OTF = $(INSTALLLIBDEST)/ebs2otf.so #ENDIF#
#TAU2EPILOG#TAU2ELG = elgconverter/tau2elg #ENDIF#
#TAU2EPILOG#INSTALLTAU2ELG = $(INSTALLDEST)/tau2elg #ENDIF#
#PERFLIB#INSTALLPERF2TAU = $(INSTALLDEST)/perf2tau #ENDIF#
#PERFLIB#PERF2TAU = perf2tau #ENDIF#
#DYNINST_STATIC_REWRITING_UNSUPPORTED#DYNINSTFLAGS=-DTAU_DYNINST_STATIC_REWRITING_UNSUPPORTED #ENDIF#
#DYNINST#DYNINC 	= -I$(DYNINSTDIR)/core/dyninstAPI/h -I$(DYNINSTDIR)/core -I$(DYNINSTDIR)/src/core/dyninstAPI/h -I$(DYNINSTDIR)/src/core -I$(DYNINSTDIR)/src/include -I$(DYNINSTDIR)/include $(DYNINSTFLAGS) #ENDIF#
#DYNINST#DYNLIBS 	= -L$(DYNINSTDIR)/lib/$(DYNINST_PLATFORM) -L$(DYNINSTDIR)/$(DYNINST_PLATFORM)/lib -ldyninstAPI -liberty -lelf #ENDIF#
#DYNINST5.2#DYNLIBS 	= -L$(DYNINSTDIR)/lib/$(DYNINST_PLATFORM) -L$(DYNINSTDIR)/src/$(DYNINST_PLATFORM)/lib -L$(DYNINSTDIR)/$(DYNINST_PLATFORM)/lib -ldyninstAPI -lelf -lcommon -lsymtabAPI $(DWARFOPTS) -ldwarf -liberty#ENDIF#
#DYNINST6#DYNLIBS 	= -L$(DYNINSTDIR)/lib/$(DYNINST_PLATFORM) -L$(DYNINSTDIR)/src/$(DYNINST_PLATFORM)/lib -L$(DYNINSTDIR)/$(DYNINST_PLATFORM)/lib -ldyninstAPI -lelf -lcommon -lsymtabAPI $(DWARFOPTS) -ldwarf -liberty -linstructionAPI #ENDIF#
#DYNINST7#DYNLIBS 	= -L$(DYNINSTDIR)/lib/$(DYNINST_PLATFORM) -L$(DYNINSTDIR)/src/$(DYNINST_PLATFORM)/lib -L$(DYNINSTDIR)/$(DYNINST_PLATFORM)/lib -ldyninstAPI -lelf -lcommon -lparseAPI -lsymtabAPI $(DWARFOPTS) -ldwarf -liberty -linstructionAPI #ENDIF#
#SP1#DYNLIBS 	= -L$(DYNINSTDIR)/$(DYNINST_PLATFORM)/lib -L$(DYNINSTDIR)/lib/$(DYNINST_PLATFORM) -ldyninstAPI #ENDIF#
#SGIMP#DYNLIBS 	= -L$(DYNINSTDIR)/$(DYNINST_PLATFORM)/lib -L$(DYNINSTDIR)/lib/$(DYNINST_PLATFORM) -ldyninstAPI $(DWARFOPTS) -ldwarf -lelf#ENDIF#
# -lelf is for Solaris only

#SOL2CC#CEXTRA=-DTAU_DOT_H_LESS_HEADERS -DSOL2CC #ENDIF#
#SUNCC#CEXTRA=-DTAU_DOT_H_LESS_HEADERS #ENDIF#


# Compiler specific flags
#KAI#COMPILEROPT = -DKAI#ENDIF#
#COMPAQ_ALPHA#COMPSPECIFIC = -DCOMPAQ_ALPHA#ENDIF#

# We must drop the $(ABI) on mips linux
PDT_ABI = $(ABI)
#MIPS32LINUX#PDT_ABI = #ENDIF#
#MIPS64LINUX#PDT_ABI = #ENDIF#

# Do we use a different compiler for PDT based tau_instrumentor?
PDT_CXX_COMP = $(CXX) $(INTELOPTS)
#PDTCXX#PDT_CXX_COMP = $(PDT_CXX) $(PDT_ABI) $(ISA) $(INTELOPTS) #ENDIF#


# Different compilation flags for PDT_CXX
#PDTNOFSTMTS#PDT_NOFSTMTS = -DPDT_NOFSTMTS#ENDIF#
#IBM_FORTRAN#PDT_OTHER_FLAGS = -DTAU_USE_SIZE_INSTEAD_OF_SIZEOF #ENDIF#
#PGI#PDT_OTHER_FLAGS = -DTAU_USE_SIZE_INSTEAD_OF_SIZEOF #ENDIF#
#IBM64_FORTRAN#PDT_OTHER_FLAGS = -DTAU_USE_SIZE_INSTEAD_OF_SIZEOF #ENDIF#
#GNU_GFORTRAN#PDT_OTHER_FLAGS = -DTAU_USE_SIZE_INSTEAD_OF_SIZEOF #ENDIF#
#NAGWARE_FORTRAN#PDT_OTHER_FLAGS = -DTAU_ALT_FORTRAN_INSTRUMENTATION #ENDIF#
PDT_COMPILE_FLAGS=$(PDT_NOFSTMTS) $(PDT_OTHER_FLAGS)
PDT_CFLAGS = $(PDTCFLAGS) $(PDT_COMPILE_FLAGS)
#PDTSGICC#PDT_CFLAGS = -LANG:std -D__SGIMP__ $(PDT_COMPILE_FLAGS) #ENDIF#
#PDTALPHACXX#PDT_CFLAGS = -x c++ -D__USE_STD_IOSTREAM $(PDT_COMPILE_FLAGS) #ENDIF#
#PDTHPACC#PDT_CFLAGS = -AA $(PDT_COMPILE_FLAGS) #ENDIF#
#DYNINST41##PKGOPT = -DTAU_DYNINST -DTAU_DYNINST41PLUS #ENDIF#

#DEBUGPROF#DEBUGOPT = -DDEBUG #ENDIF#

#GNU#GNUOPT = -DTAU_GNU -DTAU_DOT_H_LESS_HEADERS -fPIC #ENDIF#
#CRAYXMT#GNUOPT = -DTAU_GNU -DTAU_DOT_H_LESS_HEADERS -fPIC #ENDIF#

SAGE1DIR=
INCLUDEPATH = $(SAGE1DIR)/lib/include
SAGE1LIBS = $(SAGE1DIR)/lib/$(CONFIG_ARCH)

INSTALLDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin
INSTALLLIBDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib
#PDT#INSTALLINSTR 	= $(INSTALLDEST)/tau_instrumentor #ENDIF#
#PDT#INSTALLWRAP 	= $(INSTALLDEST)/tau_wrap #ENDIF#
#PDT#INSTALLPDTOMPCHECK 	= $(INSTALLDEST)/tau_ompcheck #ENDIF#
#PDT#INSTALLHEADERLIST = $(INSTALLDEST)/tau_header_list #ENDIF#
INSTALLSELECTFILE = $(INSTALLDEST)/tau_selectfile

# LIST ALL OTF-DEPENDENT UTILS HERE
INSTALL_OTF_TOOLS = $(INSTALL_EBS2OTF)

#OPARI#INSTALLTAUOMPCHECK 	= $(INSTALLPDTOMPCHECK) #ENDIF#
#DYNINST#INSTALLTAURUN 	= $(INSTALLDEST)/tau_run #ENDIF#
#DYNINST#INSTALLTAURUN 	= $(INSTALLDEST)/tau_run #ENDIF#
#TAU_LARGEFILE#LARGEFILEOPT = -DTAU_LARGEFILE -D_LARGEFILE64_SOURCE #ENDIF#
#SP1#LARGEFILEOPT = -DTAU_LARGEFILE -D_LARGEFILE64_SOURCE -D_LARGE_FILE_API #ENDIF#

CFLAGS        = $(USER_OPT) -I../include $(CEXTRA) $(CWARNING) $(COMPSPECIFIC) $(COMPILEROPT) $(PKGOPT) $(LARGEFILEOPT) $(DEBUGOPT) $(GNUOPT)

PDTCFLAGS     = $(USER_OPT) -I../include $(CEXTRA) $(CWARNING) $(COMPSPECIFIC) $(COMPILEROPT) $(PKGOPT) $(LARGEFILEOPT) $(DEBUGOPT)


HDRS	      =

MAKEFILE      = Makefile

OBJS	      = tau_reduce.o pprof.o  tau_events.o tau_merge.o tau_selective.o \
		tau_instrumentor.o tau_instrument.o user_event_data.o \
		function_data.o pprof_elem.o tau_run.o \
		tau_timecorrect.o tau_wrap.o

PRINT	      = pr

EXECUTABLES =	tau_reduce pprof tau_convert tau_merge $(RUNTIME1) $(RUNTIME2) $(RUNTIME3)\
		$(RUNTIME4) $(RUNTIME5) $(INSTR) $(TAURUN) $(TRACE2PROFILE) \
		$(TAU_TIMECORRECT) $(TAU2VTF) $(VTF2PROFILE) $(TAU2OTF) $(TAU2ELG) \
		$(TAUOMPCHECK) $(PERF2TAU) $(WRAP) $(HEADERLIST)

all:		$(EXECUTABLES)

tau_convert: 	tau_convert.c 
		$(CC) $(CFLAGS)  tau_convert.c -o tau_convert -lm $(LEXTRA)

#tau2profile:	tau2profile.o
#		$(CXX) $(CFLAGS) tau2profile.o -o tau2profile -lm $(LEXTRA) $(TAU_TRACE_INPUT_LIB)

#tau2profile.o:	tau2profile.cpp
#		$(CXX) $(CFLAGS) tau2profile.cpp -c

tau_timecorrect:	tau_timecorrect.o
		$(CXX) $(CFLAGS) tau_timecorrect.o -o tau_timecorrect -lm $(LEXTRA) $(TAU_TRACE_INPUT_LIB)

opari/opari:
		cd opari ; $(MAKE) opari

tau_timecorrect.o:	tau_timecorrect.cpp
		$(CXX) $(CFLAGS) tau_timecorrect.cpp -c

tau2vtf: 	tau2vtf.o
		$(CXX) $(CFLAGS) tau2vtf.o -o tau2vtf -lm $(LEXTRA) -L$(VTFDIR)/$(TAU_ARCH)/lib -lvtf3 $(TAU_TRACE_INPUT_LIB)

tau2vtf.o:	tau2vtf.cpp
		$(CXX) $(CFLAGS)  tau2vtf.cpp -c -I$(VTFDIR)/include 

tau2otf: 	tau2otf.o
		$(CXX) $(CFLAGS) tau2otf.o -o tau2otf -lm $(LEXTRA) $(OTFLIBS)

tau2otf.o:	tau2otf.cpp
		$(CXX) $(CFLAGS)  tau2otf.cpp -c -I$(OTFDIR)/include -I$(OTFDIR)/include/vampirtrace

ebs2otf/ebs2otf.so:
		cd ebs2otf ; $(MAKE)

trace2profile/trace2profile:
		cd trace2profile; $(MAKE) "MAKE=$(MAKE)" "CFLAGS=$(CFLAGS)"

vtfconverter/vtf2profile: 	
		cd vtfconverter; $(MAKE) "MAKE=$(MAKE)" "CFLAGS=$(CFLAGS)"

elgconverter/tau2elg:
		cd elgconverter; $(MAKE) "MAKE=$(MAKE)"

tau_merge: 	tau_merge.o tau_events.o 
		$(CXX) $(CFLAGS) tau_merge.o tau_events.o -o tau_merge -lm $(LEXTRA)

tau_merge.o: 	tau_merge.c
		$(CC) $(CFLAGS)  -c tau_merge.c

tau_events.o: 	tau_events.cpp
		$(CXX) $(CFLAGS)  -c tau_events.cpp

tau_instrumentor: tau_instrumentor.o tau_instrument.o tau_selective.o $(PDTDIR)/$(PDTARCHDIR)/lib/libpdb.a 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) tau_instrumentor.o tau_instrument.o tau_selective.o -o $@ $(PDTOPTS) -lm $(LEXTRA) 

tau_instrumentor.o: tau_instrumentor.cpp 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -I$(PDTDIR)/include -c tau_instrumentor.cpp

tau_header_list: tau_header_list.cpp
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -I$(PDTDIR)/include tau_header_list.cpp -o $@ $(PDTOPTS) -lm $(LEXTRA) 

tau_selectfile: tau_selectfile.o tau_selective_dyn.o tau_instrument.dyn.o
		$(PDT_CXX_COMP) tau_selectfile.o tau_selective_dyn.o tau_instrument.dyn.o -o $@ $(PDTOPTS) -lm $(LEXTRA) 

tau_selectfile.o: tau_selectfile.cpp
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -I$(PDTDIR)/include -c tau_selectfile.cpp 

tau_instrument.o: tau_instrument.cpp 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -I$(PDTDIR)/include -c tau_instrument.cpp

tau_instrument.dyn.o: tau_instrument.dyn.cpp 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -DTAU_DYNINST -DTAU_DYNINST41PLUS -I$(PDTDIR)/include -c tau_instrument.dyn.cpp -o tau_instrument.dyn.o

tau_selective.o: tau_selective.cpp 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -I$(PDTDIR)/include -c tau_selective.cpp

tau_selective_dyn.o: tau_selective.dyn.cpp 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -DTAU_DYNINST -DTAU_DYNINST41PLUS  -I$(PDTDIR)/include -c tau_selective.dyn.cpp -o tau_selective_dyn.o

tau_run: 	tau_run.o tau_selective_dyn.o  tau_instrument.dyn.o
		$(CXX) $(CFLAGS) tau_run.o tau_selective_dyn.o tau_instrument.dyn.o -o $@ $(DYNLIBS) $(LEXTRA) 

tau_run.o: 	tau_run.cpp
		$(CXX) $(CFLAGS) $(DYNINC) -DTAU_DYNINST -DTAU_DYNINST41PLUS -c tau_run.cpp

tau_wrap: tau_wrap.o tau_selective.o tau_instrument.o $(PDTDIR)/$(PDTARCHDIR)/lib/libpdb.a 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) tau_wrap.o tau_selective.o tau_instrument.o -o $@ $(PDTOPTS) -lm $(LEXTRA) 

tau_wrap.o: tau_wrap.cpp 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -I$(PDTDIR)/include -c tau_wrap.cpp

tau_ompcheck: tau_ompcheck.o $(PDTDIR)/$(PDTARCHDIR)/lib/libpdb.a 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) tau_ompcheck.o -o $@ $(PDTOPTS) -lm $(LEXTRA) 

tau_ompcheck.o: tau_ompcheck.cpp 
		$(PDT_CXX_COMP) $(PDT_CFLAGS) -I$(PDTDIR)/include -c tau_ompcheck.cpp

perf2tau: perf2tau.o $(PERFLIBDIR)/libperfpp.a 
		gcc perf2tau.o -o perf2tau -L$(PERFLIBDIR) -lperfpp -lm

perf2tau.o: perf2tau.c
		gcc -c perf2tau.c -I$(PERFINCDIR)


$(PDTDIR)/$(PDTARCHDIR)/lib/libpdb.a:
		@echo 
		@echo "******************************************************"
		@echo "PDT Configuration Error. $(PDTDIR)/$(PDTARCHDIR)/lib/libpdb.a not found."
	        @echo "Please configure PDT again and then install TAU."
		@echo 
		@echo "******************************************************"
	        @exit

tau_reduce:	tau_reduce.o pprof_elem.o
		$(CXX) $(CFLAGS) tau_reduce.o pprof_elem.o -o tau_reduce -lm $(LEXTRA)

tau_reduce.o: 	tau_reduce.cpp
		$(CXX) $(CFLAGS) -c tau_reduce.cpp

pprof_elem.o: 	pprof_elem.cpp
		$(CXX) $(CFLAGS) -c pprof_elem.cpp

#pprof:		pprof.cpp
#		$(CXX) $(CFLAGS)  function_data.cpp user_event_data.cpp pprof.cpp -o pprof -lm $(LEXTRA)

pprof:		pprof.o function_data.o user_event_data.o
		$(CXX) $(CFLAGS) pprof.o user_event_data.o function_data.o -o pprof -lm $(LEXTRA)

pprof.o:	pprof.cpp
		$(CXX) $(CFLAGS) -c pprof.cpp

function_data.o:	function_data.cpp
		$(CXX) $(CFLAGS) -c function_data.cpp

user_event_data.o:	user_event_data.cpp
		$(CXX) $(CFLAGS) -c user_event_data.cpp

clean:
		/bin/rm -rf *.o $(EXECUTABLES) *.dep lib*$(LSX) *.ii *.ti ti_files cxx_repository ii_files rii_files 
		cd opari ; make clean
		cd ebs2otf ; $(MAKE) clean

$(INSTALLDEST)/pprof: pprof
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_reduce: tau_reduce
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
		  else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_merge: tau_merge
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_convert: tau_convert
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/trace2profile: trace2profile/trace2profile
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_timecorrect: tau_timecorrect
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_instrumentor: tau_instrumentor
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_header_list: tau_header_list
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_selectfile: tau_selectfile
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_wrap: tau_wrap
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_ompcheck: tau_ompcheck
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_run: tau_run
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau2vtf: tau2vtf
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau2otf: tau2otf
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/vtf2profile: vtfconverter/vtf2profile
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) vtfconverter/vtf2profile $(INSTALLDEST)

$(INSTALLDEST)/opari: opari/opari
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) opari/opari $(INSTALLDEST)

$(INSTALLDEST)/tau2elg: elgconverter/tau2elg
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) elgconverter/tau2elg $(INSTALLDEST)

$(INSTALLDEST)/perf2tau: perf2tau
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) perf2tau $(INSTALLDEST)

$(INSTALLLIBDEST)/ebs2otf.so: ebs2otf/ebs2otf.so
		@echo Installing $? in $(INSTALLLIBDEST)
		if [ -d $(INSTALLLIBDEST) ] ; then true; \
                   else mkdir $(INSTALLLIBDEST) ;fi
		$(TAU_INSTALL) ebs2otf/ebs2otf.so $(INSTALLLIBDEST)
		$(TAU_INSTALL) ebs2otf/ebs2otf.pm $(INSTALLLIBDEST)
		cd ebs2otf; $(MAKE) cleanup

install:	$(INSTALLDEST)/pprof $(INSTALLDEST)/tau_reduce \
$(INSTALLDEST)/tau_merge $(INSTALLDEST)/tau_convert $(INSTALLDEST)/opari \
$(INSTALLTRACE2PROFILE) $(INSTALLTAU_TIMECORRECT) $(INSTALLPCXX) $(INSTALLINSTR) \
$(INSTALLTAURUN) $(INSTALLTAU2VTF) $(INSTALLVTF2PROFILE) $(INSTALLTAU2OTF) \
$(INSTALLTAU2ELG) $(INSTALLTAUOMPCHECK) $(INSTALLPERF2TAU) $(INSTALLWRAP) \
$(INSTALLHEADERLIST) $(INSTALLSELECTFILE) $(INSTALL_OTF_TOOLS) 

program:        $(PROGRAM)


