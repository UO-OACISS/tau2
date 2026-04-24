all: matmult

matmult: matmult.o matmult_host.o matmult_init.o
	tau_cxx.sh -L${CUDALIB} -lcudart -o $@ $^

matmult.o: matmult.cu
	nvcc --compiler-options='-finstrument-functions' -g -c $< -o $@

matmult_host.o: matmult_host.cpp
	nvcc -I. -g -c $< -o $@

matmult_init.o: matmult_init.c
	tau_cc.sh -I. -g -c $< -o $@

clean: 
	rm -rf matmult matmult.o matmult_host.o matmult_init.o  *pomp* *opari*

run:
	tau_exec -T serial,openmp,cupti -cupti ./matmult
