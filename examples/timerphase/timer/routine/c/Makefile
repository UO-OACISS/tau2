include ../../../../../include/Makefile
OPT=-optVerbose -optTauSelectFile=select.tau -optKeepFiles
CC=$(TAU_COMPILER) $(OPT) $(TAU_CC)
CXX=$(TAU_COMPILER) $(OPT) $(TAU_CXX)
F90=$(TAU_COMPILER) $(OPT) $(TAU_F90)

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
