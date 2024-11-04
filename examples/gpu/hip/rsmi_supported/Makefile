CXXFLAGS += -std=c++11 -O3 -I$(ROCM_PATH)/include
LDFLAGS += -std=c++11 -O3 -L$(ROCM_PATH)/lib -Wl,-rpath,$(ROCM_PATH)/lib -lrocm_smi64

all:  supported

HIPCC=amdclang++

supported.o : supported.cpp
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

supported: supported.o Makefile
	$(HIPCC) $(CXXFLAGS) $< -lm -o $@ $(LDFLAGS)

.PHONY: clean

clean:
	rm -f supported *.o profile*

