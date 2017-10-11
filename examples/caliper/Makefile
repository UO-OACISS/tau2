###############################################################################
## Makefile for caliper test example #
###############################################################################
TAU_MAKEFILE ?=../../include/Makefile

include $(TAU_MAKEFILE)

MAKE = make

CC = $(TAU_CC)

CFLAGS = $(TAU_INCLUDE) $(TAU_CALIPER_INCLUDE_OPTS) $(TAU_MPI_INCLUDE) $(TAU_DEFS) -fopenmp

LDFLAGS = $(TAU_MPI_LIBS) $(TAU_LIBS) -L$(CALIPERDIR)/lib64 -L$(CALIPERDIR)/lib -lcaliper -fopenmp

OBJS = cali-example.o

BINS = cali-example

all: install
	@echo

install: $(BINS)

cali-example.o: cali-example.c
	$(CC) $(CFLAGS) -c cali-example.c -o cali-example.o

cali-example: cali-example.o
	$(CC) $(LDFLAGS) cali-example.o -o cali-example

clean:
	/bin/rm -rf *.o $(BINS) profile.*
