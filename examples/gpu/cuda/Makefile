all: matmult

matmult: matmult.o
	nvcc -o $@ $<

matmult.o: matmult.cu
	nvcc -g -c $< -o $@

clean: 
	rm -rf matmult matmult.o

run:
	tau_exec -T serial -cuda ./matmult
