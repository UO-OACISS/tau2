#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.acl.lanl.gov/tau		           **
#****************************************************************************
#*    Copyright 1997-2005					   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
#######################################################################
##                  pC++/Sage++  Copyright (C) 1993,1995             ##
##  Indiana University  University of Oregon  University of Rennes   ##
#######################################################################

TAUROOTDIR	= ../..
include $(TAUROOTDIR)/include/Makefile

CHARMC = charmc
OPTS = -g

pgm: pgm.C
	$(CHARMC) $(OPTS) $(TAU_DEFS) $(TAU_INCLUDE) -c pgm.C
	$(CHARMC) $(OPTS) -o pgm pgm.o -language converse++ $(TAU_LIBS)

clean :
	$(RM) *.o pgm charmrun
