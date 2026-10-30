TAU_MAKEFILE ?=../../../../include/Makefile

include $(TAU_MAKEFILE)

CXX = TAU_MAKEFILE=$(TAU_MAKEFILE) $(TAU_PREFIX_INSTALL_DIR)/$(CONFIG_ARCH)/bin/tau_cxx.sh -optSaltInst

CXXFLAGS = -g -O2 -std=c++17

all: matmult

matmult: matmult.o
	$(CXX) $(CXXFLAGS) matmult.o -o $@

matmult.o: matmult.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	/bin/rm -f matmult matmult.o profile.* *.inst.*
