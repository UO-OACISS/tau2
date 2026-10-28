TAU_MAKEFILE ?= ../../../include/Makefile
include $(TAU_MAKEFILE)

CXX = $(TAU_CXX)
CXXFLAGS = $(TAU_INCLUDE) $(TAU_DEFS) -g
TARGET = slurm_metadata

.PHONY: all test clean

all: test

$(TARGET): slurm_metadata.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(TAU_LIBS)

test: $(TARGET)
	sh ./verify_profile.sh ./$(TARGET)

clean:
	rm -f $(TARGET) profile.0.0.0
