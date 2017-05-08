###############################################################################
## Top level Makefile for Plugins #
###############################################################################

include ../include/Makefile

PLUGININSTALLDIR = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/plugins

MAKE = make

CC = $(TAU_CC)

CXX = $(TAU_CXX)

#Defining the sub-directories that need to be made

#TUNING_POLICY=tuningpolicy
TEST_PLUGINS=testplugins
MPIT_MVAPICH_TUNING_POLICIES=MPIT_mvapich_tuning_policies

SUBDIR = $(TEST_PLUGINS) $(MPIT_MVAPICH_TUNING_POLICIES) #$(TUNING_POLICY)

all: install
	@echo

install: 
	@for i in ${SUBDIR}; do (echo "*** COMPILING $$i DIRECTORY"; cd $$i;\
             $(MAKE) "MAKE=$(MAKE)" install ) || exit $$?; done

clean:
	@for i in ${SUBDIR}; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" clean || exit 0); done

