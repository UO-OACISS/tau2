include ../../../include/Makefile
#TAU_OPENCL_INC=/opt/intel/oneapi/compiler/2021.1-beta10/linux/include/sycl
TAU_OPENCL_INC=${ROCM_PATH}/include

#TAU_OPENCL_LIB=-L$(TAU_OPENCL_INC)/../lib/x86_64 -L$(TAU_OPENCL_INC)/../lib -lOpenCL
#TAU_OPENCL_LIB=-L/opt/intel/oneapi/compiler/2021.1-beta10/linux/lib -lOpenCL
#TAU_CXX=clang++
TAU_OPENCL_LIB=-L$(TAU_OPENCL_INC)/../lib/x86_64 -L$(TAU_OPENCL_INC)/../lib -L${ROCM_PATH}/lib -Wl,-rpath,${ROCM_PATH}/lib -lamdocl64
TAU_OPENCL_LIB=$(TAU_OPENCL_LIBRARY)

all: libmatmult.so

libmatmult.so: matmult.o Makefile
	$(TAU_CXX) -g -o $@ $< $(TAU_OPENCL_LIB) -shared

matmult.o: matmult.cpp
	$(TAU_CXX) -I$(TAU_OPENCL_INC) -g -c $< -o $@ -fPIC

clean:
	rm -rf libmatmult.so matmult.o profile.*

run:
	tau_exec -T serial,cupti -opencl ./matmult
