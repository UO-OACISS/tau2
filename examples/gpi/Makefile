TOPROOT = $(PWD)

GPI_DIR = /home/gpi/

CC = gcc
CPP = g++
CC = tau_cc.sh
CPP = tau_cxx.sh

#icc
#CFLAGS = -O2 -g

#gcc
CFLAGS = -O2 -g -std=gnu99

INCLUDE = -I$(GPI_DIR)/include

#icc
#LIBS    = -i-static -static-libcxa -L$(GPI_DIR)/lib64 -libverbs15 -lGPI -Wl,-rpath=$(GPI_DIR)/lib64:

#gcc
LIBS    = -L$(GPI_DIR)/lib64 -libverbs15 -lGPI -Wl,-rpath=$(GPI_DIR)/lib64:

OBJS = gatherscatter.o

all: gatherscatter.x

gatherscatter.x: gatherscatter.o
	$(CC) $< -o $@ $(LIBS) -lMCTP1 -lrt

.c.o: $<
	$(CC) $(CFLAGS) $(INCLUDE) -o $@ $< -c

.cpp.o: $<
	$(CPP) $(CFLAGS) $(INCLUDE) -o $@ $< -c

clean:
	rm -rf *~ \#_* *.o *.x
