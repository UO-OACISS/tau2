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
 
EXE = rbc

CSRC = $(wildcard *.c)
COBJ = $(CSRC:.c=.o)

CC = tau_cc.sh
LD = tau_cc.sh

CFLAGS  = 
LDFLAGS =

MAKEFILE      	= Makefile

RM 	      	= /bin/rm -f


##############################################

all: $(EXE)

$(EXE): $(COBJ)
	$(LD) $(LDFLAGS) $(COBJ) -o $(EXE)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean: 	
	$(RM) $(COBJ) $(EXE)
