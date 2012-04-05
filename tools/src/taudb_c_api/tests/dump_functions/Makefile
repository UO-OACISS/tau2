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

include ../../Makefile.inc

HEADERS=${wildcard ../../include/*.h}
SOURCES=${wildcard *.c}
OBJS=${patsubst %.c, %.o, ${SOURCES}}

TAUDBLIB=../../libtaudb.a
LIBS=$(POSTGRESQL_LIBRARIES)
INCLUDE=-I. -I../../include $(POSTGRESQL_INCLUDES)
TARGET=../libtaudb_dump.a

all: $(TARGET)

$(TARGET): $(OBJS)
	echo "  Making : $(TARGET)"
	ar -cr $(TARGET) $(OBJS)

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
