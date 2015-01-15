all: unified_memory

unified_memory: unified_memory.o
	nvcc -o $@ $<

unified_memory.o: unified_memory.cu
	nvcc -g -c $< -o $@

clean: 
	rm -rf unified_memory unified_memory.o


