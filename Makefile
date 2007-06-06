#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/tau	           **
#****************************************************************************
#*    Copyright 1997-2000 					   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#*    Research Center Juelich, ZAM Germany                                 **
#****************************************************************************
#######################################################################
##                              TAU (C) 1996                         ##
##           based on TAU/pC++/Sage++  Copyright (C) 1993,1995       ##
##  Indiana University  University of Oregon  University of Rennes   ##
#######################################################################
 
########### Automatically modified by the configure script ############
CONFIG_ARCH=default
CONFIG_CC=gcc
CONFIG_CXX=g++
PCXX_OPT=-g
USER_OPT=-g
TAUROOT=
#######################################################################
 
include include/Makefile

#INTELCXXLIBICC#INTELOPTS = -cxxlib-icc #ENDIF#
############# Standard Defines ##############
CC = $(CONFIG_CC) $(ABI) $(ISA)
CXX = $(CONFIG_CXX) $(ABI) $(ISA) $(INTELOPTS)
TAU_INSTALL = /bin/cp
TAU_SHELL = /bin/sh
LSX = .a
#############################################
#PDT#PDTEXAMPLE = examples/autoinstrument examples/reduce #ENDIF#
#MPI#MPIEXAMPLES = examples/pi examples/NPB2.3 #ENDIF#

# Pete Beckman  (3/16/95)

# This makefile recursively calls MAKE in each subdirectory

#PTX#CC=cc#ENDIF#
LINKER	= $(CC)

# tools EVERYONE needs
#BASIC = utils src/Profile examples/instrument
BASIC = utils src/Profile 

# library and tools
EXPORTS = utils src/Profile 

# Example Programs
EXAMPLES = examples/instrument examples/threads \
examples/cthreads examples/fortran examples/f90 $(MPIEXAMPLES) $(PDTEXAMPLE)

# PC++ Support
#PCXX#PCXX=lang_support/pc++#ENDIF#

# HPC++ Support
#HPCXX#HPCXX=lang_support/hpc++#ENDIF#

# AnsiC Support
#ANSIC#ANSIC=lang_support/ansic#ENDIF#

# Trace Reader Library
#TRACE#TRACEINPUT=src/TraceInput#ENDIF#
#TRACE#TRACE2PROFILE=utils/trace2profile#ENDIF#

#PERFLIB#BASIC=utils #ENDIF#

#VTF#VTFCONVERTER=utils/vtfconverter#ENDIF#

#SLOG2#SLOGCONVERTER=utils/slogconverter/src#ENDIF#

#TAU2EPILOG#ELGCONVERTER=utils/elgconverter #ENDIF#



# Subdirectories to make resursively
SUBDIR  = $(TRACEINPUT) $(BASIC) $(PCXX) $(HPCXX) $(ANSIC) $(VTFCONVERTER) $(SLOGCONVERTER) $(ELGCONVERTER) $(TRACE2PROFILE)

all:
	@echo "At the installation root, use \"make install\" "

exports : 
	@echo "Determining Configuration..."
	@if [ x`utils/ConfigQuery -arch` = xdefault ] ; then \
          (echo Run the configure script before attempting to compile ; \
           exit 1) ; \
         else echo System previously configured as a `utils/ConfigQuery -arch` ; fi
	@echo "*********** RECURSIVELY MAKING SUBDIRECTORIES ***********"
	@for i in ${EXPORTS}; do (echo "*** COMPILING $$i DIRECTORY"; cd $$i;\
             $(MAKE) "MAKE=$(MAKE)" "CC=$(CC)" "CXX=$(CXX)" "LINKER=$(LINKER)" ); done
	@echo "***************** DONE ************************"

tests: 
	@echo "Determining Configuration..."
	@if [ x`utils/ConfigQuery -arch` = xdefault ] ; then \
          (echo Run the configure script before attempting to compile ; \
           exit 1) ; \
         else echo System previously configured as a `utils/ConfigQuery -arch` ; fi
	@echo "*********** RECURSIVELY MAKING SUBDIRECTORIES ***********"
	@for i in ${EXAMPLES}; do (echo "*** COMPILING $$i DIRECTORY"; cd $$i;\
             $(MAKE) "MAKE=$(MAKE)" "CC=$(CC)" "CXX=$(CXX)" "LINKER=$(LINKER)" ) || exit $?; done
	@echo "***************** DONE ************************"

install: .clean
	@echo "Determining Configuration..."
	@if [ x`utils/ConfigQuery -arch` = xdefault ] ; then \
          (echo Run the configure script before attempting to compile ; \
           exit 1) ; \
         else echo System previously configured as a `utils/ConfigQuery -arch` ; fi
	@echo "*********** RECURSIVELY MAKING SUBDIRECTORIES ***********"
	@for i in ${SUBDIR}; do (echo "*** COMPILING $$i DIRECTORY"; cd $$i;\
             $(MAKE) "MAKE=$(MAKE)" "CC=$(CC)" "CXX=$(CXX)" "LINKER=$(LINKER)" HOSTTYPE=$(HOSTTYPE) install) || exit $?; done
	@echo "***************** DONE ************************"

.clean:
	@for i in ${SUBDIR} ${EXAMPLES} ; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" clean); done
	touch .clean

clean:
	@for i in ${SUBDIR} ${EXAMPLES} ; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" clean); done

cleandist:	clean cleangood
cleaninstall:	clean cleangood
cleangood:
	/bin/rm -f make.log
	@echo "Deleting *~ .#* core *.a *.sl *.o *.dep"
	@find . \( -name \*~ -o -name .\#\* -o -name core \) \
	   -exec /bin/rm {} \; -print
	@find . \( -name \*.a -o -name \*.sl -o -name \*.o -o -name \*.dep \) \
	   -exec /bin/rm {} \; -print
	@if [ ! -d bin/$(CONFIG_ARCH) ] ; then true; \
	      else /bin/rm -r bin/$(CONFIG_ARCH) ; fi
	@if [ ! -d lib/$(CONFIG_ARCH) ] ; then true; \
	      else /bin/rm -r lib/$(CONFIG_ARCH) ; fi
	@grep "^#" ./build/Config.info > ./build/Config.info~~0; \
	/bin/mv ./build/Config.info~~0 ./build/Config.info

.RECURSIVE: ${SUBDIR}

${SUBDIR}: FRC
	cd $@; $(MAKE) "MAKE=$(MAKE)" "CC=$(CC)" "CXX=$(CXX)" "LINKER=$(LINKER)" all

FRC:

