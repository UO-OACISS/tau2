all: matmult

matmult: matmult.o
	nvcc -o $@ $<

matmult.o: matmult.cu
	nvcc -g -lineinfo -c $< -o $@

clean: 
	rm -rf matmult matmult.o *.csv *.txt *.cubin profile*

run:
	tau_exec -T cupti,serial -cupti -sass=kernel -csv ./matmult
