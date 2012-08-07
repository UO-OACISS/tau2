TAU_OPENCL_INC=
TAU_OPENCL_LIB=-L$(TAU_OPENCL_INC)/../lib -lOpenCL

all: matmult

matmult: matmult.o
	g++ $(TAU_OPENCL_LIB) -g -o $@ $<

matmult.o: matmult.cpp
	g++ -I$(TAU_OPENCL_INC) -g -c $< -o $@

clean: 
	rm -rf matmult matmult.o

run:
	tau_exec -T serial -opencl ./matmult
