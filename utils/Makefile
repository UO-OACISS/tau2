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
 
include ../include/Makefile
########### Automatically modified by the configure script ############
CONFIG_ARCH=default
CONFIG_CC=gcc
CONFIG_CXX=g++
PCXX_OPT=
USER_OPT=
TAU_PREFIX_INSTALL_DIR=
PDTDIR=
DYNINSTDIR=
#######################################################################
 
#MIPSR8K#ISA          = -mips4		   #ENDIF#

############# Standard Defines ##############
CC = $(CONFIG_CC) $(ABI) $(ISA)
CXX = $(CONFIG_CXX) $(ABI) $(ISA)
INSTALL = /bin/cp
SHELL = /bin/sh
LSX = .a
#############################################

LINKER	= $(CC)

LEXTRA = 
#HP_CFLAGS#CEXTRA = -Ae#ENDIF#
#HP_CFLAGS#LSX = .sl#ENDIF#
#SYMMETRY#LEXTRA = -lseq#ENDIF#
#PTX#LEXTRA = -lseq#ENDIF#
#SP1#CEXTRA = -D__SP1__ -D_POSIX_SOURCE -D_XOPEN_SOURCE #ENDIF#
#USE_IBMXLC#CEXTRA = -DTAU_DOT_H_LESS_HEADERS -DTAU_XLC #ENDIF#
#USE_DECCXX#CEXTRA = -DTAU_DOT_H_LESS_HEADERS -x c++ #ENDIF#
#CM5#CEXTRA = -D__CM5__#ENDIF#
#SGIMP#CEXTRA = -D__SGIMP__#ENDIF#
#PDTSGICC#CEXTRA = -LANG:std -D__SGIMP__#ENDIF#
#PCXX#RUNTIME1 = pcxx_merge#ENDIF#
#PCXX#RUNTIME2 = pcxx_convert#ENDIF#
#PCXX#RUNTIME3 = pcxx_dump#ENDIF#
#HPCXX#RUNTIME1 = pcxx_merge#ENDIF#
#HPCXX#RUNTIME2 = pcxx_convert#ENDIF#
#HPCXX#RUNTIME3 = pcxx_dump#ENDIF#
#HPCXX#RUNTIME4 = pcxx2ar#ENDIF#
#HPCXX#RUNTIME5 = ar2pcxx#ENDIF#
WARNINGSOPT	= -w
#CRAYCC#WARNINGSOPT= -DCRAYCC  #ENDIF#
#PGI1.7#LEXTRA = -lstd -lstrm  #ENDIF#
#PGI#CEXTRA = -DPGI -DTAU_DOT_H_LESS_HEADERS  #ENDIF#
#FUJITSU#CEXTRA = -DFUJITSU -DTAU_DOT_H_LESS_HEADERS  #ENDIF#
CWARNING	= $(WARNINGSOPT)
#PDT#PDTOPTS 	= -I$(PDTDIR)/include -L$(PDTDIR)/$(CONFIG_ARCH)/lib -lpdb #ENDIF#
#PDT#INSTR 		= tau_instrumentor #ENDIF#
#DYNINST#TAURUN 	= tau_run #ENDIF#
#DYNINST#DYNINC 	= -I$(DYNINSTDIR)/core/dyninstAPI/h #ENDIF#
#DYNINST#DYNLIBS 	= -L$(DYNINSTDIR)/lib/$(DYNINST_PLATFORM) -ldyninstAPI -liberty -lelf #ENDIF#
#SGIMP#DYNLIBS 	= -L$(DYNINSTDIR)/lib/$(DYNINST_PLATFORM) -ldyninstAPI -ldwarf -lelf#ENDIF#
# -lelf is for Solaris only

#SOL2CC#CEXTRA=-DTAU_DOT_H_LESS_HEADERS -DSOL2CC #ENDIF#


# Compiler specific flags
#KAI#COMPILEROPT = -DKAI#ENDIF#
#COMPAQ_ALPHA#COMPSPECIFIC = -DCOMPAQ_ALPHA#ENDIF#


SAGE1DIR=
INCLUDEPATH = $(SAGE1DIR)/lib/include
SAGE1LIBS = $(SAGE1DIR)/lib/$(CONFIG_ARCH)

INSTALLDEST	= $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin
#PDT#INSTALLINSTR 	= $(INSTALLDEST)/tau_instrumentor #ENDIF#
#DYNINST#INSTALLTAURUN 	= $(INSTALLDEST)/tau_run #ENDIF#

CFLAGS        = $(USER_OPT) -I../include $(CEXTRA) $(CWARNING) $(COMPSPECIFIC) $(COMPILEROPT)

HDRS	      =

EVHEADER      = $(SAGE1DIR)/tulip/include/pcxx_events_def.h \
		$(SAGE1DIR)/tulip/include/pcxx_events.h \
                $(SAGE1DIR)/tulip/include/pcxx_ansi.h \
		$(SAGE1DIR)/tulip/include/pcxx_machines.h

LDFLAGS	      = $(PCXX_OPT)

MAKEFILE      = Makefile

OBJS	      = pprof.o  tau_events.o tau_merge.o

PRINT	      = pr

SRCS	      = pcxx_merge.c \
		pcxx_convert.c

EXECUTABLES =	pprof tau_convert tau_merge $(RUNTIME1) $(RUNTIME2) $(RUNTIME3)\
		$(RUNTIME4) $(RUNTIME5) $(INSTR) $(TAURUN)

all:		$(EXECUTABLES)

pcxx_merge:	pcxx_merge.c $(EVHEADER)
		$(LINKER) -D__PCXX__ $(LDFLAGS) $(CEXTRA) -I$(SAGE1DIR)/tulip/include \
		pcxx_merge.c -o pcxx_merge $(LEXTRA)

pcxx_convert:	pcxx_convert.c $(EVHEADER)
		$(LINKER) -D__PCXX__ $(LDFLAGS) $(CEXTRA) -I$(SAGE1DIR)/tulip/include \
		pcxx_convert.c -o pcxx_convert

pcxx2ar : pcxx2ar.c $(EVHEADER)
		$(LINKER)  -D__PCXX__ $(LDFLAGS) $(CEXTRA) \
		   -I$(SAGE1DIR)/tulip/include \
		   pcxx2ar.c -o pcxx2ar

tau_convert: 	tau_convert.c 
		$(CXX) $(CFLAGS)  tau_convert.c -o tau_convert -lm $(LEXTRA)

tau_merge: 	tau_merge.o tau_events.o
		$(CXX) $(CFLAGS)  tau_merge.o tau_events.o -o tau_merge -lm $(LEXTRA)

tau_merge.o: 	tau_merge.c
		$(CXX) $(CFLAGS)  -c tau_merge.c

tau_events.o: 	tau_events.cpp
		$(CXX) $(CFLAGS)  -c tau_events.cpp

tau_instrumentor: 	tau_instrumentor.cpp  $(PDTDIR)/$(CONFIG_ARCH)/lib/libpdb.a 
		$(CXX) $(CFLAGS) tau_instrumentor.cpp -o $@ $(PDTOPTS) -lm $(LEXTRA) 

tau_run: 	tau_run.cpp  
		$(CXX) $(CFLAGS) $(DYNINC) tau_run.cpp -o $@ $(DYNLIBS) $(LEXTRA) 

$(PDTDIR)/$(CONFIG_ARCH)/lib/libpdb.a:
		@echo 
		@echo "******************************************************"
		@echo "PDT Configuration Error. $(PDTDIR)/$(CONFIG_ARCH)/lib/libpdb.a not found."
	        @echo "Please configure PDT again and then install TAU."
		@echo 
		@echo "******************************************************"
	        @exit

ar2pcxx : ar2pcxx.c 
		$(LINKER) ar2pcxx.c -o $@

pprof:		pprof.cpp
		$(CXX) $(CFLAGS)  pprof.cpp -o pprof -lm $(LEXTRA)

clean:
		/bin/rm -f $(OBJS) $(EXECUTABLES) *.dep lib*$(LSX)

index:
		ctags -wx $(HDRS) $(SRCS)

$(INSTALLDEST)/pprof: pprof
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_merge: tau_merge
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_convert: tau_convert
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_instrumentor: tau_instrumentor
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(INSTALL) $? $(INSTALLDEST)

$(INSTALLDEST)/tau_run: tau_run
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(INSTALL) $? $(INSTALLDEST)

#$(INSTALLDEST)/$(RUNTIME1): $(RUNTIME1)
#		@echo Installing $? in $(INSTALLDEST)
#		if [ -d $(INSTALLDEST) ] ; then true; \
#                   else mkdir $(INSTALLDEST) ;fi
#		$(INSTALL) $? $(INSTALLDEST)
#
#$(INSTALLDEST)/$(RUNTIME2): $(RUNTIME2)
#		@echo Installing $? in $(INSTALLDEST)
#		if [ -d $(INSTALLDEST) ] ; then true; \
#                   else mkdir $(INSTALLDEST) ;fi
#		$(INSTALL) $? $(INSTALLDEST)
#
#$(INSTALLDEST)/$(RUNTIME3): $(INSTALLDEST)/$(RUNTIME2)
#		@echo Installing $? in $(INSTALLDEST)
#		if [ -d $(INSTALLDEST) ] ; then true; \
#                   else mkdir $(INSTALLDEST) ;fi
#		rm -f $(INSTALLDEST)/$(RUNTIME3)
#		ln $(INSTALLDEST)/$(RUNTIME2) $(INSTALLDEST)/$(RUNTIME3)
#
#$(INSTALLDEST)/$(RUNTIME4): $(RUNTIME4)
#		@echo Installing $? in $(INSTALLDEST)
#		if [ -d $(INSTALLDEST) ] ; then true; \
#                   else mkdir $(INSTALLDEST) ;fi
#		$(INSTALL) $? $(INSTALLDEST)
#
#$(INSTALLDEST)/$(RUNTIME5): $(RUNTIME5)
#		@echo Installing $? in $(INSTALLDEST)
#		if [ -d $(INSTALLDEST) ] ; then true; \
#                   else mkdir $(INSTALLDEST) ;fi
#		$(INSTALL) $? $(INSTALLDEST)
#
#PCXX#INSTALLPCXX = $(INSTALLDEST)/$(RUNTIME1) $(INSTALLDEST)/$(RUNTIME2) $(INSTALLDEST)/$(RUNTIME3) $(INSTALLDEST)/$(RUNTIME4) $(INSTALLDEST)/$(RUNTIME5)#ENDIF#
#HPCXX#INSTALLPCXX = $(INSTALLDEST)/$(RUNTIME1) $(INSTALLDEST)/$(RUNTIME2) $(INSTALLDEST)/$(RUNTIME3) $(INSTALLDEST)/$(RUNTIME4) $(INSTALLDEST)/$(RUNTIME5)#ENDIF#

install:	$(INSTALLDEST)/pprof $(INSTALLDEST)/tau_merge $(INSTALLDEST)/tau_convert $(INSTALLPCXX) $(INSTALLINSTR) $(INSTALLTAURUN)

program:        $(PROGRAM)


