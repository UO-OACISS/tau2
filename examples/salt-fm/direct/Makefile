TAU_MAKEFILE ?=../../../include/Makefile

include $(TAU_MAKEFILE)

CC = $(TAU_CC)

CXX = $(TAU_CXX)

CFLAGS = -g -O2 $(TAU_INCLUDE) $(TAU_DEFS)

LIBS = $(TAU_LIBS)

all: matmult

matmult.inst.c: matmult.c
	saltfm $< -o $@ --

matmult.o: matmult.inst.c
	$(CC) $(CFLAGS) -c $< -o $@

matmult: matmult.o
	$(CXX) -g matmult.o -o $@ $(LIBS)

clean:
	/bin/rm -f matmult matmult.o matmult.inst.c profile.*
