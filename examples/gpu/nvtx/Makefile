NVCC := "$(CUDAROOT)/bin/nvcc"

INCLUDES = -I$(CUDAROOT)/include/nvtx3

manual_nvtx: manual_nvtx.o  
	$(NVCC) -o $@ $^  $(INCLUDES) -DUSE_NVTX

manual_nvtx.o: manual_nvtx.cu
	$(NVCC) -c $(INCLUDES) -DUSE_NVTX $<

clean:
	rm -f profile.* manual_nvtx manual_nvtx.o
