TAU_MAKEFILE ?=../../../include/Makefile

CC = tau_cc.sh -optCompInst -optShared -tau_makefile=$(TAU_MAKEFILE) -fopenmp

default: libexamples.so

libexamples.so: examples.o
	$(CC) --shared -o $@ $^

examples.o: examples.c examples.h
	$(CC) -c $< -fPIC

clean:
	rm -f *.o *.so
