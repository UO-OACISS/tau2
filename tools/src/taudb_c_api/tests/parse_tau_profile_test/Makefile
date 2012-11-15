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

PWD=$(shell pwd)
ROOTDIR?=$(PWD)/../..
include $(ROOTDIR)/Makefile.in

HEADERS=${wildcard *.h}
SOURCES=${wildcard *.c}
OBJS=${patsubst %.c, %.o, ${SOURCES}}

TARGET=taudb_test
TAUDBLIB=$(ROOTDIR)/lib/libtaudb.a
DUMP_FUNCS=../libtaudb_dump.a
INCLUDE=-I. -I$(ROOTDIR)/include $(INCLUDES) -I../dump_functions

all: $(TARGET)

$(TARGET): $(TAUDBLIB) $(HEADERS) $(OBJS)
	echo " Linking : $@"
	$(CC) $(FLAGS) $@.o $(DUMP_FUNCS) -o $@ $(TAUDBLIB) $(LIBS)

# rebuild everything if the headers or the makefile changes
$(OBJS): $(HEADERS) Makefile $(ROOTDIR)/Makefile.in

.c.o:
	echo "Building : $@ from $<"
	$(CC) $(FLAGS) -c $(INCLUDE) $< -o $@ $(PERFDMF_SUPPORT) $(DEBUG)

test: $(TARGET)
	echo " Testing : $(TARGET)..."
	#time `./$(TARGET) $(shell pwd) >& $(TARGET).log`
	time $(shell ./$(TARGET) $(PWD) >& $(PWD)/$(TARGET).log)
	echo " \n...Done. Contents of $(PWD)/$(TARGET).log: \n"
	cat $(PWD)/$(TARGET).log
	echo ""

clean:
	echo "Removing : $(OBJS) $(TARGET)"
	/bin/rm -f $(OBJS) $(TARGET) $(TARGET).log *.pdb *.inst.c

remake: clean all
