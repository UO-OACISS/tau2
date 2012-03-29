#****************************************************************************
#*                      TAU Performance System (R)                          *
#*                      http://tau.uoregon.edu                              *
#****************************************************************************
#*    Copyright 2008                                                       **
#*    Department of Computer and Information Science, University of Oregon **
#*    Juelich Supercomputing Center, NIC, Germany                          ** 
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
# Use gmake to build matmult

.SUFFIXES : .c
.SILENT :

CC=gcc

HEADERS=${wildcard *.h}
SOURCES=${wildcard *.c}
OBJS=${patsubst %.c, %.o, ${SOURCES}}

TARGET=taudb_test
LIBS=-lpq
INCLUDE=-I.
FLAGS=-g
PERFDMF_SUPPORT=-DTAUDB_PERFDMF
DEBUG=-DTAUDB_DEBUG

all: $(TARGET)

$(TARGET): $(OBJS) $(HEADERS)
	echo " Linking : $(TARGET)"
	$(CC) $(FLAGS) $(OBJS) -o $(TARGET) $(LIBS)

# rebuild everything if the headers or the makefile changes
$(OBJS): $(HEADERS) Makefile

.c.o:
	echo "Building : $@ from $<"
	$(CC) $(FLAGS) -c $(INCLUDE) $< -o $@ $(PERFDMF_SUPPORT) $(DEBUG)

test: $(TARGET)
	./$(TARGET)

clean:
	echo "Removing : $(OBJS) $(TARGET)"
	/bin/rm -f $(OBJS) $(TARGET)

remake: clean all
