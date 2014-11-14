all: sass_source

sass_source: sass_source.o
	#nvcc -lineinfo -o $@ $< -L/usr/local/packages/cuda/6.5/extras/CUPTI/lib64 -lcupti
	nvcc --compiler-options=-finstrument-functions -lineinfo -o $@ $< -L/usr/local/packages/cuda/7.0/extras/CUPTI/lib64 -lcupti

sass_source.o: sass_source.cu
	#nvcc -lineinfo -g -c $< -o $@
	nvcc --compiler-options=-finstrument-functions -lineinfo -g -c $< -o $@

clean: 
	rm -rf sass_source sass_source.o

run:
	tau_exec -T serial -cuda ./sass_source

