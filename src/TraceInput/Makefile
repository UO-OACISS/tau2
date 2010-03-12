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

##############################################
PRELINK_PHASE = $(CXX) --prelink_objects $(OBJS) #ENDIF##PGICC#
PRELINK_PHASE = #ENDIF##PGINOPRELINK#
#############################################


#USE_DECCXX#EXTRAOBJS=cxx_repository/*.o #ENDIF#


RM		= /bin/rm -rf

INSTALLDEST     = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib

#PERFLIB#TAU_INCLUDE     = -I$(TAU_PREFIX_INSTALL_DIR)/include #ENDIF#
#PERFLIB#TAU_DEFS     = $(TAUDEFS) #ENDIF#

CFLAGS          = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE) $(USER_OPT)

CXXFLAGS        = $(TAU_INCLUDE) $(TAU_DEFS) $(TAU_MPI_INCLUDE) $(USER_OPT) -I.

LIBS            = $(TAU_MPI_LIBS) $(TAU_SHLIBS) 

LDFLAGS	      	= $(USER_OPT) $(TAU_LDFLAGS)

LIBTARGET 	= libTAU_traceinput$(TAU_CONFIG).a

OBJS		= TAU_tf.o TAU_Cwrapper.o TAU_tf_writer.o

HDRS 		= ../../include/TAU_tf.h ../../include/TAU_tf_writer.h TAU_tf_decl.h TAU_tf_headers.h

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
	$(PRELINK_PHASE)
	$(AR) $(ARFLAGS) $(LIBTARGET) $(OBJS) $(EXTRAOBJS)
	$(TAU_RANLIB) $(LIBTARGET)

TAU_tf.o : TAU_tf.cpp $(HDRS)
	$(TAU_CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TAU_tf_writer.o : TAU_tf_writer.cpp $(HDRS)
	$(TAU_CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TAU_Cwrapper.o : TAU_Cwrapper.cpp $(HDRS)
	$(TAU_CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

clean: 	
	$(RM) $(OBJS) $(LIBTARGET) *.ii *.ti ti_files cxx_repository ii_files rii_files 
##############################################
#/***************************************************************************
# * $RCSfile: Makefile,v $   $Author: wspear $
# * $Revision: 1.12 $   $Date: 2010/03/12 23:11:20 $
# * TAU_VERSION_ID: $Id: Makefile,v 1.12 2010/03/12 23:11:20 wspear Exp $ 
# ***************************************************************************/
