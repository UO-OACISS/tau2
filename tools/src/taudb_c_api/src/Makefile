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

ROOTDIR?=$(shell pwd)/..
include $(ROOTDIR)/Makefile.in

HEADERS=${wildcard $(ROOTDIR)/include/*.h}
SOURCES=${wildcard */*.c}
OBJS=${patsubst %.c, %.o, ${SOURCES}}
TARGET=$(ROOTDIR)/lib/libtaudb.a

# rebuild everything if the headers or the makefile changes
all: $(TARGET)

$(TARGET): $(OBJS)
	echo "  Making : $@"
	rm -f $(TARGET)
	$(AR) -qc $(TARGET) $(OBJS)

$(OBJS): Makefile $(ROOTDIR)/Makefile.in $(HEADERS) $(SOURCES)
	mkdir -p $(ROOTDIR)/lib
	@for i in database memory parsers analysis ; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" "ROOTDIR=$(ROOTDIR)" || exit 0); done

clean:
	@for i in database memory parsers analysis ; do (cd $$i; $(MAKE) "MAKE=$(MAKE)" "ROOTDIR=$(ROOTDIR)" clean || exit 0); done
	echo "Removing $(TARGET)"
	rm -f $(TARGET)


remake: clean all
