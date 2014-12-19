all: unified_memory

# unified_memory: unified_memory.o
# 	nvcc -o $@ $< -L/usr/local/packages/cuda/6.5/extras/CUPTI/lib64 -lcupti

unified_memory: unified_memory.o
	nvcc -o $@ $<

unified_memory.o: unified_memory.cu
	nvcc -g -c $< -o $@

clean: 
	rm -rf unified_memory unified_memory.o

run:
	tau_exec -T serial -cuda ./unified_memory

