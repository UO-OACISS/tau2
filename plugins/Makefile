###############################################################################
## Top level Makefile for Plugins #
###############################################################################

include ../include/Makefile

PLUGININSTALLDIR = $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/lib/shared$(TAU_CONFIG)

MAKE = make

CC = $(TAU_CC)

CXX = $(TAU_CXX)

#Defining the sub-directories that need to be made

#TUNING_POLICY=tuningpolicy
TEST_PLUGINS=testplugins
EXAMPLES=examples
LLVM=llvm

SUBDIR = $(TEST_PLUGINS) $(EXAMPLES) #$(TUNING_POLICY)

all: install
	@echo

install: 
	@for i in ${SUBDIR}; do (echo "*** COMPILING $$i DIRECTORY"; cd $$i;\
             $(MAKE) "MAKE=$(MAKE)" install ) || exit $$?; done

# Compile the LLVM one separately
llvm:
	cd $(LLVM) && $(MAKE) "MAKE=$(MAKE)" install ) || exit $$?

clean:
	@for i in ${SUBDIR}; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" clean || exit 0); done

