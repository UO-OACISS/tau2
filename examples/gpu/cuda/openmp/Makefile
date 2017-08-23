CC=nvcc
CFLAGS=-Xcompiler -fopenmp 
LIBS=-lgomp 
all: add

add: add.o
	$(CC) -o $@ $< $(LIBS)

add.o: add.cu
	$(CC) -g -c $< -o $@ $(CFLAGS)

clean: 
	rm -rf add add.o

run:
	tau_exec -T serial,cupti,openmp -cupti ./add
