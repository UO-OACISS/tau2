CC=nvcc
all: matmult

matmult: matmult.o
	$(CC) -o $@ $< --default-stream per-thread

matmult.o: matmult.cu
	$(CC) -g -c $< -o $@ --default-stream per-thread

clean: 
	rm -rf matmult matmult.o

run:
	tau_exec -T serial -cuda ./matmult
