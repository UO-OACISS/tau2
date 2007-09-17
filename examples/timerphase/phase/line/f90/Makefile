include ../../../../../include/Makefile
OPT=-optVerbose -optTauSelectFile=select.tau -optKeepFiles
CC=$(TAU_COMPILER) $(OPT) $(TAU_CC)
CXX=$(TAU_COMPILER) $(OPT) $(TAU_CXX)
F90=$(TAU_COMPILER) $(OPT) $(TAU_F90)

all: foo 

foo: foo.o f1.o
	$(F90) foo.o f1.o -o foo
foo.o: foo.f90
	$(F90) -c foo.f90
f1.o: f1.c
	$(CC) -c f1.c
clean:
	/bin/rm -f foo *.o *.inst.* *.pdb profile.*
run: foo
	./foo
	pprof 
	cat select.tau
