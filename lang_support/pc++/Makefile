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
#######################################################################
 
############# Standard Defines ##############
CC = $(CONFIG_CC)
CXX = $(CONFIG_CXX)
INSTALL = /bin/cp
SHELL = /bin/sh
LSX = .a
#############################################

INSTALLDEST = ../../bin/$(CONFIG_ARCH)

SUBDIR1 = breezy cgm
SUBDIR  = ${SUBDIR1}

all:
	@for i in ${SUBDIR1}; do (echo "*** COMPILING $$i DIRECTORY"; cd $$i;\
	  $(MAKE) "MAKE=$(MAKE)" "CC=$(CC)" "CXX=$(CXX)" "LINKER=$(LINKER)" $@); done

clean:
	@for i in ${SUBDIR1}; do (cd $$i;\
	  $(MAKE) "MAKE=$(MAKE)" $@); done

cleaninstall: clean

install: 
	@for i in ${SUBDIR1}; do (echo "*** COMPILING $$i DIRECTORY"; cd $$i;\
	  $(MAKE) "MAKE=$(MAKE)" "CC=$(CC)" "CXX=$(CXX)" "LINKER=$(LINKER)" $@); done


${SUBDIR}: FRC
	cd $@; $(MAKE) "MAKE=$(MAKE)" "CC=$(CC)" "CXX=$(CXX)" "LINKER=$(LINKER)" all

FRC:

