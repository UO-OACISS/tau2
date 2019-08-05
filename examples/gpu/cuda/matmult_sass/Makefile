all: matmult

matmult: matmult.o
	nvcc -o $@ $<

matmult.o: matmult.cu
	nvcc -g -lineinfo -c -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_60,code=sm_60 $< -o $@

clean: 
	rm -rf matmult matmult.o *.csv *.txt *.cubin profile*

run:
	tau_exec -T cupti,serial -cupti -sass=kernel -csv ./matmult
