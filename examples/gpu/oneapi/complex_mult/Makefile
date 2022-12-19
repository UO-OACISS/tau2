CXX = dpcpp
#CXX = clang++
CXXFLAGS = -std=c++17 -g -o
LDFLAGS =


#CXXFLAGS = -fsycl-targets=nvptx64-nvidia-cuda -o
#CXXFLAGS = -fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfx90a -o
EXE_NAME = complex_mult.exe
SOURCES = src/complex_mult.cpp

all: main

main:
	$(CXX) $(CXXFLAGS) $(EXE_NAME) $(SOURCES) $(LDFLAGS)

run: 
	./$(EXE_NAME)

clean: 
	rm -rf $(EXE_NAME) profile.* MULT* *.trc *.edf traces*
