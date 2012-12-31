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

HEADERS=${wildcard *.h}
SOURCES=${wildcard *.c}
OBJS=${patsubst %.c, %.o, ${SOURCES}}

TARGET=taudb_query_stats_test
TAUDBLIB=../../lib/libtaudb.a
DUMP_FUNCS=../libtaudb_dump.a
INCLUDE=-I. -I../../include $(INCLUDES) -I../dump_functions

all: $(TARGET)

$(TARGET): $(TAUDBLIB) $(HEADERS) $(OBJS) $(DUMP_FUNCS)
	echo " Linking : $@"
	$(CC) $(FLAGS) $@.o $(DUMP_FUNCS) -o $@ $(TAUDBLIB) $(LIBS)

# rebuild everything if the headers or the makefile changes
$(OBJS): $(HEADERS) Makefile ../../Makefile.in

.c.o:
	echo "Building : $@ from $<"
	$(CC) $(FLAGS) -c $(INCLUDE) $< -o $@ $(PERFDMF_SUPPORT) $(DEBUG)

test: $(TARGET)
	echo " Testing : $(TARGET)..."
	./$(TARGET) $(TEST_CONFIG) $(TEST_TRIAL_ID)
	# time `./$(TARGET) $(TEST_CONFIG) $(TEST_TRIAL_ID) >& $(TARGET).log`
	# echo " \n...Done. Contents of $(TARGET).log: \n"
	# cat $(TARGET).log
	echo ""

clean:
	echo "Removing : $(OBJS) $(TARGET)"
	/bin/rm -f $(OBJS) $(TARGET) $(TARGET).log *.pdb *.inst.c

remake: clean all
