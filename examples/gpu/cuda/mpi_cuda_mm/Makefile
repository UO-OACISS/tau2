MPI_CC=mpicc
CC=nvcc
CFLAGS=-Xcompiler -fopenmp -Xcompiler -fPIC
LIBS=-lgomp 
all: add


add: libadd.so driver.o
	$(MPI_CC) -o $@ driver.o -L. -ladd -Wl,-rpath,`pwd` $(LIBS) -fopenmp

add.o: add.cu
	$(CC) -g -c $< -o $@ $(CFLAGS) 

libadd.so: add.o
	$(CC) -shared -o libadd.so add.o

driver.o: driver.c
	$(MPI_CC) -c $< -o $@ -fopenmp
clean: 
	rm -rf add add.o libadd.so driver.o  profile* traces* MULT*

run: add
	mpirun -np 2 tau_exec -T ompt,mpi,pdt,papi,cupti,openmp -ompt -cupti ./add

run_uninst: add
	mpirun -np 2 ./add
