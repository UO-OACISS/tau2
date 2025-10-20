#****************************************************************************
#*                      TAU Portable Profiling Package                     **
#*                      http://www.cs.uoregon.edu/research/paracomp/tau    **
#****************************************************************************
#*    Copyright 2025                                                       **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
include ../../include/Makefile

LIB = libperfetto.a

SRCS = perfetto.cc
OBJS = $(SRCS:.cc=.o)

PERFETTO_CXXFLAGS = -std=c++17 -fPIC -O3 -DNDEBUG

CXXFLAGS = $(TAU_CXXFLAGS) $(PERFETTO_CXXFLAGS)

all: $(LIB)

$(LIB): $(OBJS)
	$(AR) rcs $@ $^

%.o: %.cc
	$(CXX) $(CXXFLAGS) -I. -c $< -o $@

clean:
	rm -f $(OBJS) $(LIB)
