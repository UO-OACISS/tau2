#****************************************************************************
#*			TAU Portable Profiling Package			   **
#*			http://www.cs.uoregon.edu/research/tau	           **
#****************************************************************************
#*    Copyright 1997  						   	   **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
#****************************************************************************
#**      File            : Makefile                                        **
#**      Description     : TAU Callstack Tracing Package. The UG Senior    **
#**			   year project of Mike Kaufman. The csUI tool is  **
#**			   written with XForms which needs to be installed **
#**			   http://bragg.phys.uwm.edu/xforms	           **
#**      Author          : Mike Kaufman	                                   **
#**      Contact         : {mikek,sameer}@cs.uoregon.edu       		   **
#**      Flags           : Compile with -DDEBUG for debugging info         **
#**      Documentation   : See http://www.cs.uoregon.edu/research/tau       **
#****************************************************************************

########### Automatically modified by the configure script ############
CONFIG_ARCH=default
CONFIG_CC=gcc
CONFIG_CXX=g++
PCXX_OPT=-g
USER_OPT=-g
TAUROOT=
XFORMSROOT=/local/apps/xforms/FORMS
#######################################################################
 
#MIPSR8K##ABI 	     = -64		   #ENDIF#
#MIPSR8K#ISA          = -mips4		   #ENDIF#
############# Standard Defines ##############
CC = $(CONFIG_CC) $(ABI) $(ISA)
CXX = $(CONFIG_CXX) $(ABI) $(ISA)
DEFINES = 
TAU_INSTALL = /bin/cp
TAU_SHELL = /bin/sh
LSX = .a
#############################################

INSTALLDEST	= $(TAUROOT)/bin/$(CONFIG_ARCH)

LIBS = -L$(XFORMSROOT) -lforms -lX11 -lm
INCLUDE = -I$(XFORMSROOT) -I$(TAUROOT)/include
OBJS = csUI.o csUI_cb.o 
TARGET     = csUI
OPTIONS = -O -w
############################################

all: $(TARGET)

install: $(INSTALLDEST)/$(TARGET)

$(TARGET):	$(OBJS)
	$(CXX) $(OPTIONS) -o $(TARGET) $(OBJS) $(LIBS) $(DEFINES)

.C.o:
	$(CXX) $(INCLUDE) $(OPTIONS)  -c $< -o $@ $(DEFINES)


#############################################

clean:
	rm -f $(OBJS) $(TARGET)
	
$(INSTALLDEST)/$(TARGET): $(TARGET)
		@echo Installing $? in $(INSTALLDEST)
		if [ -d $(INSTALLDEST) ] ; then true; \
                   else mkdir $(INSTALLDEST) ;fi
		$(TAU_INSTALL) $? $(INSTALLDEST)

