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

include ../Makefile.inc

HEADERS=${wildcard ../include/*.h}
SOURCES=${wildcard *.c}
OBJS=${patsubst %.c, %.o, ${SOURCES}}

TARGET=../lib/libtaudb.a

all: $(TARGET)

$(TARGET): $(OBJS) $(HEADERS)
	echo "  Making : $@"
	rm -f $(TARGET)
	$(AR) -qc $(TARGET) $(OBJS)

# rebuild everything if the headers or the makefile changes
$(OBJS): $(HEADERS) Makefile ../Makefile.inc

.c.o:
	echo "Building : $@ from $<"
	$(CC) $(FLAGS) -c -I../include $(INCLUDE) $< -o $@ $(DEBUG)

clean:
	echo "Removing : $(OBJS) $(TARGET)"
	/bin/rm -f $(OBJS) $(TARGET) *.pdb *.inst.c

test:

remake: clean all
