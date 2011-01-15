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

#CRAYXMT#CC = $(TAU_CC_FE) $(ABI) $(ISA) #ENDIF#
#CRAYXMT#CXX = $(TAU_CXX_FE) $(ABI) $(ISA) #ENDIF#

#CRAYCNL#CC = $(TAU_CC_FE) $(ABI) $(ISA) #ENDIF#
#CRAYCNL#CXX = $(TAU_CXX_FE) $(ABI) $(ISA) #ENDIF#

AR		= $(TAU_AR)
#CRAYXMT#AR		= /usr/bin/ar  #ENDIF#
#CRAYCNL#AR		= /usr/bin/ar  #ENDIF#

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
#PGICC#PRELINK_PHASE = $(CXX) --prelink_objects $(OBJS) #ENDIF#
#PGINOPRELINK#PRELINK_PHASE = #ENDIF#
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
	$(TAU_RANLIB_FE) $(LIBTARGET)

TAU_tf.o : TAU_tf.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TAU_tf_writer.o : TAU_tf_writer.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

TAU_Cwrapper.o : TAU_Cwrapper.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $(@:.o=.cpp)

clean: 	
	$(RM) $(OBJS) $(LIBTARGET) *.ii *.ti ti_files cxx_repository ii_files rii_files 
##############################################
#/***************************************************************************
# * $RCSfile: Makefile,v $   $Author: sameer $
# * $Revision: 1.14 $   $Date: 2010/04/07 22:33:51 $
# * TAU_VERSION_ID: $Id: Makefile,v 1.14 2010/04/07 22:33:51 sameer Exp $ 
# ***************************************************************************/
