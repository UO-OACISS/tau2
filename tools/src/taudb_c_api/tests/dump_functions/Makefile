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

include ../../Makefile.in

HEADERS=${wildcard ../../include/*.h}
SOURCES=${wildcard *.c}
OBJS=${patsubst %.c, %.o, ${SOURCES}}

TAUDBLIB=../../lib/libtaudb.a
INCLUDE=-I. -I../../include $(INCLUDES)
TARGET=../libtaudb_dump.a

all: $(TARGET)

$(TARGET): $(OBJS)
	echo "  Making : $(TARGET)"
	$(AR) -qc $(TARGET) $(OBJS)
	$(RANLIB) $(TARGET)

# rebuild everything if the headers or the makefile changes
$(OBJS): $(HEADERS) Makefile

.c.o:
	echo "Building : $@ from $<"
	$(CC) $(FLAGS) -c $(INCLUDE) $< -o $@ $(PERFDMF_SUPPORT) $(DEBUG)

clean:
	echo "Removing : $(OBJS)"
	/bin/rm -f $(OBJS) $(TARGET) *.pdb *.inst.c

test:

remake: clean all
