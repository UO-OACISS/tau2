CC=nvcc

all: matmult

matmult: matmult.o
	$(CC) -o $@ $<

matmult.o: matmult.cu
	$(CC) -g --compiler-options=-finstrument-functions -c $< -o $@

clean: 
	rm -rf matmult matmult.o

run:
	tau_exec -T serial -cuda ./matmult
