#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.acl.lanl.gov/tau		           **
#****************************************************************************
#*    Copyright 1997  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
#######################################################################
##                              TAU (C) 1996                         ##
##           based on TAU/pC++/Sage++  Copyright (C) 1993,1995       ##
##  Indiana University  University of Oregon  University of Rennes   ##
#######################################################################
 
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
INSTALL = /bin/cp
SHELL = /bin/sh
LSX = .a
#############################################

# Pete Beckman  (3/16/95)

# This makefile recursively calls MAKE in each subdirectory

#PTX#CC=cc#ENDIF#
LINKER	= $(CC)

# tools EVERYONE needs
BASIC = utils src/Profile examples/instrument

# Example Programs
EXAMPLES = examples/matrix examples/instrument examples/pi

# PC++ Support
#PCXX#PCXX=lang_support/pc++#ENDIF#

# HPC++ Support
#HPCXX#HPCXX=lang_support/hpc++#ENDIF#

# AnsiC Support
#ANSIC#ANSIC=lang_support/ansic#ENDIF#


# Subdirectories to make resursively
SUBDIR  = $(BASIC) $(PCXX) $(HPCXX) $(ANSIC)

all:
	@echo "At the installation root, use \"make install\" "

examples : 
	@echo "*********** RECURSIVELY MAKING SUBDIRECTORIES ***********"
	@for i in ${EXAMPLES}; do (echo "*** COMPILING $$i DIRECTORY"; cd $$i;\
             $(MAKE) "MAKE=$(MAKE)" "CC=$(CC)" "CXX=$(CXX)" "LINKER=$(LINKER)" ); done
	@echo "***************** DONE ************************"

install:
	@echo "Determining Configuration..."
	@if [ x`utils/ConfigQuery -arch` = xdefault ] ; then \
          (echo Run the configure script before attempting to compile ; \
           exit 1) ; \
         else echo System previously configured as a `utils/ConfigQuery -arch` ; fi
	@echo "*********** RECURSIVELY MAKING SUBDIRECTORIES ***********"
	@for i in ${SUBDIR}; do (echo "*** COMPILING $$i DIRECTORY"; cd $$i;\
             $(MAKE) "MAKE=$(MAKE)" "CC=$(CC)" "CXX=$(CXX)" "LINKER=$(LINKER)" install); done
	@echo "***************** DONE ************************"

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

