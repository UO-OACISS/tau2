NVIDIA = /usr/local/cuda-7.0
CUPTI_LIB = $(NVIDIA)/extras/CUPTI/lib64

all: sass_source

sass_source: sass_source.o
	nvcc -lineinfo -o $@ $< #-L$(CUPTI_LIB) -lcupti

sass_source.o: sass_source.cu
	nvcc -lineinfo -g -c $< -o $@

clean: 
	rm -rf sass_source sass_source.o

run:
	tau_exec -T serial -cuda ./sass_source

