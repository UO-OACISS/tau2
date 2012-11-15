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

ROOTDIR?=$(shell pwd)/../..
MAKEFILE_INC=$(ROOTDIR)/Makefile.in
include $(MAKEFILE_INC)

HEADERS=${wildcard $(ROOTDIR)/include/*.h}
SOURCES=${wildcard *.c}
OBJS=${patsubst %.c, %.o, ${SOURCES}}
TARGET=$(OBJS)

all: $(TARGET)

# rebuild everything if the headers or the makefile changes
$(OBJS): $(HEADERS) Makefile $(MAKEFILE_INC)

.c.o:
	echo "Building : $@ from $<"
	$(CC) $(FLAGS) -c -I$(ROOTDIR)/include $(INCLUDE) $< -o $@ $(DEBUG)

clean:
	echo "Removing : $(OBJS) $(TARGET)"
	/bin/rm -f $(OBJS) $(TARGET) *.pdb *.inst.c

test:

remake: clean all
