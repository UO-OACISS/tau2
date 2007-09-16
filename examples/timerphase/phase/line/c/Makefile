include ../../../../../include/Makefile
OPT=-optVerbose -optTauSelectFile=select.tau -optKeepFiles
CC=$(TAU_COMPILER) $(OPT) icc
CXX=$(TAU_COMPILER) $(OPT) icpc
F90=$(TAU_COMPILER) $(OPT) ifort

all: foo 

foo: foo.o 
	$(CC) foo.o -o foo
foo.o: foo.c
	$(CC) -c foo.c
clean:
	/bin/rm -f foo *.o *.inst.* *.pdb profile.*
run: foo
	./foo
	pprof 
	cat select.tau
