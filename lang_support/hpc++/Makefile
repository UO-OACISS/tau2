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
USER_OPT=
#######################################################################
 
############# Standard Defines ##############
CC = $(CONFIG_CC)
CXX = $(CONFIG_CXX)
INSTALL = /bin/cp
SHELL = /bin/sh
LSX = .a
#############################################

INSTALLDEST = ../../bin/$(CONFIG_ARCH)

#SOL2#CPPCGM=hpcxx2dep#ENDIF#
SUBDIR1 = $(CPPCGM)
SUBDIR  = ${SUBDIR1}

all:
	@for i in dummy ${SUBDIR1}; do \
	if [ "$$i" != "dummy" ]; then \
	(echo "*** COMPILING $$i DIRECTORY"; cd $$i;\
	  $(MAKE) "MAKE=$(MAKE)" "CC=$(CC)" "CXX=$(CXX)" "LINKER=$(LINKER)" $@); fi ; done

clean:
	@for i in dummy ${SUBDIR1}; do \
	if [ "$$i" != "dummy" ]; then \
	(cd $$i; $(MAKE) "MAKE=$(MAKE)" $@); fi ; done

cleaninstall: clean

install: cppcgm
	@for i in dummy ${SUBDIR1}; do \
	if [ "$$i" != "dummy" ]; then \
	(echo "*** COMPILING $$i DIRECTORY"; cd $$i;\
	  $(MAKE) "MAKE=$(MAKE)" "CC=$(CC)" "CXX=$(CXX)" "LINKER=$(LINKER)" $@); fi ; done
	@@echo Installing cppcgm in $(INSTALLDEST)
	@if [ -d $(INSTALLDEST) ] ; then true; \
	  else mkdir $(INSTALLDEST) ;fi
	@$(INSTALL) cppcgm $(INSTALLDEST)


${SUBDIR}: FRC
	cd $@; $(MAKE) "MAKE=$(MAKE)" "CC=$(CC)" "CXX=$(CXX)" "LINKER=$(LINKER)" all

FRC:

