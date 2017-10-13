###############################################################################
## Makefile for caliper test example #
###############################################################################
#TAU_MAKEFILE ?=../../include/Makefile

#include $(TAU_MAKEFILE)

MAKE = make

CC = mpicc

CALIPERDIR = /home/users/sramesh/CALIPER_TAU_INTEGRATION/CALIPER

CFLAGS = -I$(CALIPERDIR)/include -fopenmp

LDFLAGS = -L$(CALIPERDIR)/lib -L$(CALIPERDIR)/lib64 -lcaliper -fopenmp

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
