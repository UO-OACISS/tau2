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
 
TAUROOTDIR	= ../../..

include $(TAUROOTDIR)/include/Makefile

CC		= $(TAU_COMPILER) -optKeepFiles -optTau="-rv MacroVoidReturn -rn MacroFunctionReturn" $(TAU_CC)

MAKEFILE      	= Makefile

PRINT	      	= pr

RM 	      	= /bin/rm -f

TARGET	      	= ring

EXTRAOBJS     	= 

##############################################
# Original Rules
##############################################
all:		$(TARGET)	

$(TARGET):	$(TARGET).o
	$(CC) $(LDFLAGS) $(TARGET).o -o $@ $(LIBS)

$(TARGET).o : $(TARGET).c
	$(CC) $(CFLAGS) -c $(TARGET).c

clean:
	$(RM) $(TARGET) $(TARGET).o $(TARGET).pdb $(TARGET).inst.c profile.* 

##############################################
