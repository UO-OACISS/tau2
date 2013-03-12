#****************************************************************************
#*                      TAU Performance System (R)                          *
#*                      http://tau.uoregon.edu                              *
#****************************************************************************
#*    Copyright 2012                                                       **
#*    Department of Computer and Information Science, University of Oregon **
#*    Juelich Supercomputing Center, NIC, Germany                          ** 
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************

.SUFFIXES : .c

include Makefile.in

ROOTDIR=$(shell pwd)

all: Makefile Makefile.in
	mkdir -p $(ROOTDIR)/lib
	@for i in src tests ; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" "ROOTDIR=$(ROOTDIR)" || exit 0); done

test: all $(TARGET)
	@for i in tests ; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" "ROOTDIR=$(ROOTDIR)" test || exit 0); done

clean:
	@for i in src tests ; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" "ROOTDIR=$(ROOTDIR)" clean || exit 0); done

remake: clean all
