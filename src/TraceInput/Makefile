#/****************************************************************************
#**			TAU Portable Profiling Package			   **
#**			http://www.cs.uoregon.edu/research/paracomp/tau    **
#*****************************************************************************
#**    Copyright 2003  						   	   **
#**    Department of Computer and Information Science, University of Oregon **
#**    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#**    Research Center Juelich, Germany                                     **
#****************************************************************************/
#/***************************************************************************
#**	File 		: makefile 					  **
#**	Description 	: TAU trace format reader makefile                **
#**	Author		: Sameer Shende					  **
#**	Contact		: sameer@cs.uoregon.edu 	                  **
#***************************************************************************/
 
TAUROOTDIR=../../

include $(TAUROOTDIR)/include/Makefile

CXX		= $(TAU_CXX)

CC		= $(TAU_CC)

AR		= $(TAU_AR)

ARFLAGS         = rcv 

############# For KAI KCC compiler, replace AR accordingly #######
#KAI#AR                 = $(CXX)                           #ENDIF#
#KAI#ARFLAGS                    = --one_per -o                     #ENDIF#
##################################################################

############# For SGI CC compiler, replace AR accordingly #######
#SGICC#AR                       = $(CXX) -ar               #ENDIF#
#SGICC#ARFLAGS                  = -o                       #ENDIF#
##################################################################

############# For FUJITSU FCC compiler, replace AR accordingly #######
#FUJITSU#AR                     = $(CXX) --ar              #ENDIF#
#FUJITSU#ARFLAGS                = -o                       #ENDIF#
##################################################################

############# For SUN CC compiler, replace AR accordingly #######
#SOL2CC#AR                      = $(CXX) -xar              #ENDIF#
#SOL2CC#ARFLAGS                 = -o                       #ENDIF#
##################################################################


RM		= /bin/rm -f

INSTALLDEST     = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE) 

CXXFLAGS        = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE) -I.

LIBS            = $(TAU_MPI_LIBS) $(TAU_SHLIBS) 

LDFLAGS	      	= $(USER_OPT) $(TAU_LDFLAGS)

LIBTARGET 	= libTAU_traceinput$(TAU_OPTIONS).a

OBJS		= TAU_tf.o TAU_Cwrapper.o

HDRS 		= TAU_tf.h TAU_tf_decl.h TAU_tf_headers.h

##############################################

all: 	$(INSTALLDEST)/$(LIBTARGET)

install: $(INSTALLDEST)/$(LIBTARGET)

$(INSTALLDEST)/$(LIBTARGET): $(LIBTARGET)
	@echo Installing $? in $(INSTALLDEST)
	@if [ -d $(INSTALLDEST) ] ; then true; \
	  else mkdir $(INSTALLDEST) ;fi
	$(TAU_INSTALL) $? $(INSTALLDEST)
	$(RM) $(INSTALLDEST)/libTAU_tf.a
	cd $(INSTALLDEST) && ln -s $(LIBTARGET) libTAU_tf.a


$(LIBTARGET):	$(OBJS)
	$(AR) $(ARFLAGS) $(LIBTARGET) $(OBJS)
	$(TAU_RANLIB) $(LIBTARGET)

%.o : %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

clean: 	
	$(RM) $(OBJS) $(LIBTARGET) 
##############################################
#/***************************************************************************
# * $RCSfile: Makefile,v $   $Author: sameer $
# * $Revision: 1.2 $   $Date: 2003/11/16 00:21:37 $
# * TAU_VERSION_ID: $Id: Makefile,v 1.2 2003/11/16 00:21:37 sameer Exp $ 
# ***************************************************************************/
