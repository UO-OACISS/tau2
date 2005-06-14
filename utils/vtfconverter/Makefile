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
 
include ../../include/Makefile
########### Automatically modified by the configure script ############



#SGICC#LEXTRA = -lCio #ENDIF#




CONFIG_ARCH=default
CONFIG_CC=gcc
CONFIG_CXX=g++
CXX=$(CONFIG_CXX)
HEADERS=
VTFDIR=
LIBS=-L$(VTFDIR)/$(TAU_ARCH)/lib  -lvtf3 $(LEXTRA)
VTFINCLUDES=-I$(VTFDIR)/include
RM=/bin/rm -f

#The following are variables for all the files and their associated variations.
BINARYTREE=binarytree.o binarytree.*.~*
NODE=node.o node.*.~*
TRAY=tray.o tray.*.~*
NODEC=nodeC.o nodeC.*.~*
STACK=stack.o stack.*.~*
READVTF=readVTF.o readVTF.*.~*
CONVERTER=converter.o converter.*.~* vtf2profile
MAKEFILE=Makefile.~*

all:	vtf2profile

install:	vtf2profile

vtf2profile: node.o tray.o nodeC.o stack.o binarytree.o readVTF.o converter.o
	$(CXX) $(CFLAGS) -o $@ node.o tray.o nodeC.o stack.o binarytree.o readVTF.o converter.o $(LIBS)

node.o:	node.h node.cc
	$(CXX) $(CFLAGS) $(VTFINCLUDES) -c node.cc

tray.o: tray.h tray.cc
	$(CXX) $(CFLAGS) $(VTFINCLUDES) -c tray.cc

nodeC.o: nodeC.h nodeC.cc
	$(CXX) $(CFLAGS) $(VTFINCLUDES) -c nodeC.cc

stack.o: tray.h stack.h stack.cc
	$(CXX) $(CFLAGS) $(VTFINCLUDES) -c stack.cc

binarytree.o:	node.h nodeC.h binarytree.h binarytree.cc
	$(CXX) $(CFLAGS) $(VTFINCLUDES) -c binarytree.cc

readVTF.o: node.h binarytree.h readVTF.h readVTF.cc
	$(CXX) $(CFLAGS) $(VTFINCLUDES) -c readVTF.cc 

converter.o: readVTF.h converter.cc
	$(CXX) $(CFLAGS) $(VTFINCLUDES) -c converter.cc

clean:
	$(RM) $(BINARYTREE) $(NODE) $(TRAY) $(NODEC) $(STACK) $(READVTF) $(CONVERTER) $(MAKEFILE)
