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
.SILENT :

include Makefile.inc

TARGET=lib/libtaudb.a

# rebuild everything if the headers or the makefile changes
all: Makefile Makefile.inc
	mkdir -p lib
	@for i in src tests ; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" || exit 0); done

test: all $(TARGET)
	@for i in tests ; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" test || exit 0); done

clean:
	@for i in src tests ; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" clean || exit 0); done

remake: clean all
