MPI_CC=mpicc
CC=nvcc
CFLAGS=-Xcompiler -fPIC
LIBS=
all: add


add: libadd.so driver.o
	$(MPI_CC) -o $@ driver.o -L. -ladd -Wl,-rpath,`pwd` $(LIBS) 

add.o: add.cu
	$(CC) -g -c $< -o $@ $(CFLAGS) 

libadd.so: add.o
	$(CC) -shared -o libadd.so add.o

driver.o: driver.c
	$(MPI_CC) -c $< -o $@ 
clean: 
	rm -rf add add.o libadd.so driver.o  profile* traces* MULT*

run: add
	mpirun -np 2 tau_exec -T mpi,pdt,papi,cupti -cupti ./add

run_uninst: add
	mpirun -np 2 ./add
