include ../../../include/Makefile
#TAU_OPENCL_INC=/usr/local/packages/cuda/7.0/include
TAU_OPENCL_LIB=-L$(TAU_OPENCL_INC)/../lib/x86_64 -L$(TAU_OPENCL_INC)/../lib -lOpenCL
CXX=$(TAU_CXX)

all: matmult

matmult: matmult.o
	$(TAU_CXX) $(TAU_OPENCL_LIB) -g -o $@ $<

matmult.o: matmult.cpp
	$(TAU_CXX) -I$(TAU_OPENCL_INC) -g -c $< -o $@

clean: 
	rm -rf matmult matmult.o

run:
	tau_exec -T serial,cupti -opencl ./matmult
