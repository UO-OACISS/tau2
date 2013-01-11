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

# memalign and pvalloc tests are disabled by default
# Uncomment this line to enable those tests
#EXTRA_TEST_FLAGS = -DHAVE_MEMALIGN -DHAVE_PVALLOC

# Enable to test overrun 
EXTRA_TEST_FLAGS = -DTEST_OVERRUN

# Enable to test underrun 
#EXTRA_TEST_FLAGS = -DTEST_UNDERRUN

EXE = rbc

SRC = $(wildcard *.c)
OBJ = $(SRC:.c=.o)

CC = tau_cc.sh
LD = tau_cc.sh

CPPFLAGS = $(EXTRA_TEST_FLAGS)
CFLAGS  = -O2 -g
LDFLAGS =

RM 	      	= /bin/rm -f


##############################################

all: $(EXE)

run: $(EXE)
	TAU_MEMDBG_PROTECT_ABOVE=1 tau_exec -memory -T serial $(EXE)

$(EXE): $(OBJ)
	$(LD) $(LDFLAGS) $(OBJ) -o $(EXE)

%.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

clean: 	
	$(RM) $(OBJ) $(EXE)
	$(RM) *.inst.* *.pdb
