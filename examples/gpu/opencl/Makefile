include ../../../include/Makefile
#TAU_OPENCL_INC=/opt/intel/oneapi/compiler/2021.1-beta10/linux/include/sycl

TAU_OPENCL_LIB=-L$(TAU_OPENCL_INC)/../lib/x86_64 -L$(TAU_OPENCL_INC)/../lib -lOpenCL
#TAU_OPENCL_LIB=-L/opt/intel/oneapi/compiler/2021.1-beta10/linux/lib -lOpenCL
TAU_CXX=clang++

all: matmult

matmult: matmult.o
	$(TAU_CXX) $(TAU_OPENCL_LIB) -g -o $@ $<

matmult.o: matmult.cpp
	$(TAU_CXX) -I$(TAU_OPENCL_INC) -g -c $< -o $@

clean: 
	rm -rf matmult matmult.o

run:
	tau_exec -T serial,cupti -opencl ./matmult
