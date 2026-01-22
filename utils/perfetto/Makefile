#****************************************************************************
#*                      TAU Portable Profiling Package                     **
#*                      http://www.cs.uoregon.edu/research/paracomp/tau    **
#****************************************************************************
#*    Copyright 2025                                                       **
#*    Department of Computer and Information Science, University of Oregon **
#*    Advanced Computing Laboratory, Los Alamos National Laboratory        **
#****************************************************************************
#
# NOTE: This Makefile is provided for reference/standalone builds only.
# During TAU builds with -perfetto enabled, perfetto.cc is compiled directly
# into the TAU library via src/Profile/Makefile (see Perfetto.o target).
# This Makefile is not used by the main TAU build process.
#

include ../../include/Makefile

SRCS = perfetto.cc
OBJS = $(SRCS:.cc=.o)

PERFETTO_CXXFLAGS = -std=c++17 -fPIC -DNDEBUG
CXXFLAGS = $(TAU_CXXFLAGS) $(PERFETTO_CXXFLAGS) -O2 -fno-inline 

all: Perfetto.o

Perfetto.o: $(OBJS)
	$(LD) -r -o $@ $^

%.o: %.cc
	$(CXX) $(CXXFLAGS) -I. -c $< -o $@

clean:
	rm -f $(OBJS) Perfetto.o
