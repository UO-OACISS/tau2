#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/tau	           **
#****************************************************************************
#*    Copyright 1997  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
#######################################################################
##                  pC++/Sage++  Copyright (C) 1993,1995             ##
##  Indiana University  University of Oregon  University of Rennes   ##
#######################################################################
 

TAU_MAKEFILE ?=../../include/Makefile

include $(TAU_MAKEFILE)

CC		= TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_cc.sh

PRINT	      	= pr

RM 	      	= /bin/rm -rf

TARGET	      	= foo

OBJS     	= foo.o

##############################################

all:		$(TARGET)	

install: 	$(TARGET)

$(TARGET):	$(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@ $(LIBS)

foo.o: foo.c
	$(CC) $(CFLAGS) -c $< -o $@

clean: 	
	$(RM) $(OBJS) $(TARGET) profile.* *.trc *.edf *.z MULT* *.inst.* *.pdb Comp_gnu.o
##############################################
