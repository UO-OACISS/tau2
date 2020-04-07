CUDA_ARCH_FLAGS := -arch=sm_70
CC_FLAGS += $(CUDA_ARCH_FLAGS)

#EXE = dataElem_um dataElem_um_c++_1 dataElem_um_c++_2
EXE = dataElem_um

all: $(EXE)

% : %.cu
	#nvcc $< $(CC_FLAGS) $(LIB_FLAGS) -o $@ -g -O0 -I$(CUDA)/extras/CUPTI/include -L $(CUDA)/extras/CUPTI/lib64 -lcupti
	nvcc $< $(CC_FLAGS) $(LIB_FLAGS) -o $@ -g -O0

clean:
	rm -f $(EXE) profile.*